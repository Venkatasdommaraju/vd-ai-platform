from app.graph import run, build_graph, route_after_critic


def test_graph_returns_grounded_answer():
    state = run("What retrieval backend does the service use and how is it pluggable?")
    assert state["answer"], "expected a non-empty answer"
    assert state["sources"], "expected at least one cited source"
    assert state["revisions"] >= 1


def test_graph_terminates_within_revision_cap():
    state = run("Explain the planner critic loop and how it avoids infinite loops.")
    # max_revisions is 2, so analyst runs at most 3 times total
    assert state["revisions"] <= 3


def test_router_finalizes_when_no_revision_needed():
    assert route_after_critic({"needs_revision": False, "revisions": 1}) == "synthesize"


def test_router_can_loop_back():
    assert route_after_critic({"needs_revision": True, "revisions": 1}) == "analyst"


def test_graph_compiles():
    assert build_graph() is not None
