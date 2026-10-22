"""Tests for do() what-if interventions. No DETR/BERT download needed:
the encoders are replaced with small stubs."""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

import models.scm_vlm as scm_vlm_module
from models.do_intervention import (
    ClassSteerer,
    DoInterventionReasoner,
    Intervention,
    descendants_of,
    parse_do_spec,
    parse_whatif,
    resolve_label,
)

LABELS = ["person", "bicycle", "umbrella", "car", "car", "traffic light"]
DETR_LABELS = ["person", "bicycle", "car", "motorcycle", "bus", "traffic light", "umbrella", "dog"]


class StubVision(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()

    def forward(self, image_paths):
        raise RuntimeError("not used in tests")


class StubText(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()

    def forward(self, questions):
        if isinstance(questions, str):
            questions = [questions]
        g = torch.Generator().manual_seed(len(questions[0]))
        return torch.randn(len(questions), 768, generator=g)


@pytest.fixture()
def reasoner(monkeypatch):
    monkeypatch.setattr(scm_vlm_module, "VisionEncoder", StubVision)
    monkeypatch.setattr(scm_vlm_module, "TextEncoder", StubText)
    torch.manual_seed(0)
    model = scm_vlm_module.SCMVLM(num_classes=10).eval()
    model.graph.threshold = 0.0   # keep top-k edges so the graph is not empty
    classifier = nn.Linear(256, len(DETR_LABELS))
    steerer = ClassSteerer(classifier, {name: i for i, name in enumerate(DETR_LABELS)})
    return DoInterventionReasoner(model, {i: f"ans{i}" for i in range(10)}, steerer)


@pytest.fixture()
def world(reasoner):
    torch.manual_seed(1)
    return reasoner.observe_from_tensors(torch.randn(len(LABELS), 256), LABELS, "What is the person holding?")


# ---------------------------------------------------------------- parsing
@pytest.mark.parametrize("text,kind,target,new,question", [
    ("What if there were no umbrella, what is the person holding?", "remove", "umbrella", None,
     "what is the person holding?"),
    ("What if the bicycle were a motorcycle? What is the man riding?", "replace", "the bicycle",
     "motorcycle", "what is the man riding?"),
    ("What if we removed the cars - is the road busy?", "remove", "the cars", None, "is the road busy?"),
    ("What if there was a dog?", "add", None, "dog", None),
    ("what if the umbrella is replaced with a bicycle, what is he holding", "replace", "the umbrella",
     "a bicycle", "what is he holding?"),
])
def test_parse_whatif(text, kind, target, new, question):
    intervention, q = parse_whatif(text)
    assert intervention.kind == kind
    assert intervention.target == target
    assert intervention.new_label == new
    assert q == question


def test_parse_do_spec():
    assert parse_do_spec("remove:car").kind == "remove"
    spec = parse_do_spec("replace:#2->dog")
    assert spec.nodes == [2] and spec.new_label == "dog"


def test_resolve_label():
    assert resolve_label("the cars", DETR_LABELS) == "car"
    assert resolve_label("a man", DETR_LABELS) == "person"
    assert resolve_label("bike", DETR_LABELS) == "bicycle"
    assert resolve_label("traffic lights", DETR_LABELS) == "traffic light"
    assert resolve_label("spaceship", DETR_LABELS) is None


# ---------------------------------------------------------------- graph semantics
def test_descendants_follow_message_direction():
    # A[i, j] > 0 means j -> i.  Chain 0 -> 1 -> 2, node 3 isolated.
    a = torch.zeros(4, 4)
    a[1, 0] = 1.0
    a[2, 1] = 1.0
    assert descendants_of(a, [0]) == [1, 2]
    assert descendants_of(a, [2]) == []


def test_replace_cuts_parents_and_keeps_non_descendants(reasoner, world):
    result = reasoner.intervene(world, Intervention("replace", target="umbrella", new_label="dog"))
    k = LABELS.index("umbrella")
    assert torch.all(result.counterfactual.adjacency[k] == 0)               # Eq. 12: incoming cut
    other_rows = [i for i in range(len(LABELS)) if i != k]
    assert torch.equal(result.counterfactual.adjacency[other_rows], world.adjacency[other_rows])
    assert result.counterfactual.labels[k] == "dog"
    assert result.steering[k]["converged"]
    for i in result.unaffected:                                             # Eq. 14
        assert torch.equal(result.counterfactual.hidden[i], world.hidden[i])
    assert result.non_descendant_max_drift < 1e-5
    for d in result.descendants:
        assert d not in result.unaffected


def test_remove_all_instances(reasoner, world):
    result = reasoner.intervene(world, Intervention("remove", target="cars"))
    assert result.intervened_nodes == [3, 4]
    assert "car" not in result.counterfactual.labels
    assert result.counterfactual.adjacency.shape == (4, 4)
    assert result.non_descendant_max_drift < 1e-5


def test_add_creates_exogenous_node(reasoner, world):
    result = reasoner.intervene(world, Intervention("add", new_label="dog"))
    n = len(LABELS)
    assert result.counterfactual.labels[-1] == "dog"
    assert torch.all(result.counterfactual.adjacency[n] == 0)               # no parents
    assert result.non_descendant_max_drift < 1e-5


def test_no_intervention_effect_without_descendants_is_consistent(reasoner, world):
    summary = reasoner.summarise(reasoner.intervene(world, Intervention("remove", nodes=[0])))
    assert 0.0 <= summary["total_variation"] <= 1.0
    assert summary["intervened_nodes"] == ["0:person"]


def test_missing_object_raises(reasoner, world):
    with pytest.raises(ValueError):
        reasoner.intervene(world, Intervention("remove", target="bus"))


# ---------------------------------------------------------------- attribute interventions
@pytest.mark.parametrize("text,target,value,question", [
    ("what if the traffic light turns green", "the traffic light", "green", None),
    ("What if the light turned red, should the cyclist stop?", "the light", "red", "should the cyclist stop?"),
    ("what if the car was blue", "the car", "blue", None),
])
def test_parse_attribute(text, target, value, question):
    intervention, q = parse_whatif(text)
    assert (intervention.kind, intervention.target, intervention.new_label, q) == ("attribute", target, value, question)


def test_attribute_default_question():
    intervention, _ = parse_whatif("what if the traffic light turns green")
    assert intervention.default_question() == "What color is the traffic light?"


def test_attribute_intervention_keeps_class_and_cuts_parents(reasoner, world):
    result = reasoner.intervene(world, Intervention("attribute", target="the light", new_label="green"))
    k = LABELS.index("traffic light")
    assert result.intervened_nodes == [k]
    assert result.counterfactual.labels[k] == "green traffic light"
    assert torch.all(result.counterfactual.adjacency[k] == 0)
    info = result.steering[k]
    assert info["similarity_after"] > info["similarity_before"]
    assert result.non_descendant_max_drift < 1e-5


def test_attribute_that_is_a_class_becomes_replace(reasoner, world):
    result = reasoner.intervene(world, Intervention("attribute", target="bicycle", new_label="motorcycle"))
    assert result.counterfactual.labels[LABELS.index("bicycle")] == "motorcycle"
    assert result.steering[LABELS.index("bicycle")].get("attribute") is None


# ---------------------------------------------------------------- bare "what if" (no question)
def test_effect_report_lists_dependents(reasoner, world):
    # pick a node that has children in the learned graph
    parent = next(j for j in range(len(LABELS)) if (world.adjacency[:, j] > 0).any())
    result = reasoner.intervene(world, Intervention("remove", nodes=[parent]))
    effect = reasoner.effect_report(result)
    assert effect["affected_objects"], "a node with children must affect them"
    assert all(item["relative_change"] > 0 for item in effect["affected_objects"])
    assert any(item["direct"] for item in effect["affected_objects"])
    assert "influence" in effect["conclusion"]
    assert LABELS[parent] not in effect["scene_after"] or LABELS.count(LABELS[parent]) > 1


def test_effect_report_without_dependents(reasoner, world):
    leaf = next((j for j in range(len(LABELS)) if not (world.adjacency[:, j] > 0).any()), None)
    if leaf is None:
        pytest.skip("every node has children in this random graph")
    effect = reasoner.effect_report(reasoner.intervene(world, Intervention("remove", nodes=[leaf])))
    assert effect["affected_objects"] == []
    assert "nothing else in the scene changes" in effect["conclusion"]
