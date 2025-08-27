import pytest

from tfworker.custom_types import TerraformAction, TerraformStage
from tfworker.exceptions import HandlerError
from tfworker.handlers.base import BaseHandler
from tfworker.handlers.collection import HandlersCollection

ORDER = []


@pytest.fixture(autouse=True)
def clear_order():
    ORDER.clear()
    # reset singleton to ensure fresh handlers for each test
    HandlersCollection._instance = None


class DummyHandler(BaseHandler):
    actions = [TerraformAction.PLAN]

    def __init__(self):
        self._ready = True

    def execute(self, action, stage, deployment, definition, working_dir, result=None):
        ORDER.append(self.tag)


class HandlerA(DummyHandler):
    tag = "a"
    default_priority = {TerraformAction.PLAN: 50}


class HandlerB(DummyHandler):
    tag = "b"
    default_priority = {TerraformAction.PLAN: 60}
    dependencies = {TerraformAction.PLAN: {TerraformStage.POST: ["a"]}}


class HandlerC(DummyHandler):
    tag = "c"
    default_priority = {TerraformAction.PLAN: 40}


class HandlerD(DummyHandler):
    tag = "d"
    default_priority = {TerraformAction.PLAN: 60}
    dependencies = {TerraformAction.PLAN: {TerraformStage.POST: ["missing"]}}


class DummyDef:
    name = "def"


def test_dependency_and_priority_order():
    h = HandlersCollection(
        {
            "a": HandlerA(),
            "b": HandlerB(),
            "c": HandlerC(),
        }
    )
    h.exec_handlers(TerraformAction.PLAN, TerraformStage.POST, "dep", DummyDef(), ".")
    assert ORDER == ["c", "a", "b"]


def test_missing_dependency_ignored():
    h = HandlersCollection(
        {
            "a": HandlerA(),
            "d": HandlerD(),
        }
    )
    h.exec_handlers(TerraformAction.PLAN, TerraformStage.POST, "dep", DummyDef(), ".")
    assert ORDER == ["a", "d"]


class HandlerWithPlan(DummyHandler):
    tag = "with_plan"

    def has_plan(self, definition):
        return True


class HandlerWithoutPlan(DummyHandler):
    tag = "without_plan"

    def has_plan(self, definition):
        return False


class TestCheckPlanConflicts:
    def test_no_conflicts_with_no_handlers_having_plans(self):
        h = HandlersCollection(
            {
                "a": HandlerWithoutPlan(),
                "b": HandlerWithoutPlan(),
            }
        )
        # Should not raise
        h.check_plan_conflicts(DummyDef())

    def test_no_conflicts_with_one_handler_having_plan(self):
        h = HandlersCollection(
            {
                "a": HandlerWithPlan(),
                "b": HandlerWithoutPlan(),
            }
        )
        # Should not raise
        h.check_plan_conflicts(DummyDef())

    def test_conflict_with_multiple_handlers_having_plans(self):
        h = HandlersCollection(
            {
                "a": HandlerWithPlan(),
                "b": HandlerWithPlan(),
            }
        )
        with pytest.raises(
            HandlerError, match="Multiple handlers claim to have a plan"
        ):
            h.check_plan_conflicts(DummyDef())

    def test_not_ready_handler_ignored_in_conflict_check(self):
        handler_not_ready = HandlerWithPlan()
        handler_not_ready._ready = False

        h = HandlersCollection(
            {
                "a": HandlerWithPlan(),
                "b": handler_not_ready,
            }
        )
        # Should not raise since handler b is not ready
        h.check_plan_conflicts(DummyDef())


class TestHasAvailablePlan:
    def test_returns_false_when_no_handlers_have_plans(self):
        h = HandlersCollection(
            {
                "a": HandlerWithoutPlan(),
                "b": HandlerWithoutPlan(),
            }
        )
        assert h.has_available_plan(DummyDef()) is False

    def test_returns_true_when_handler_has_plan(self):
        h = HandlersCollection(
            {
                "a": HandlerWithPlan(),
                "b": HandlerWithoutPlan(),
            }
        )
        assert h.has_available_plan(DummyDef()) is True

    def test_raises_on_conflict(self):
        h = HandlersCollection(
            {
                "a": HandlerWithPlan(),
                "b": HandlerWithPlan(),
            }
        )
        with pytest.raises(
            HandlerError, match="Multiple handlers claim to have a plan"
        ):
            h.has_available_plan(DummyDef())

    def test_ignores_not_ready_handlers(self):
        handler_not_ready = HandlerWithPlan()
        handler_not_ready._ready = False

        h = HandlersCollection(
            {
                "a": HandlerWithoutPlan(),
                "b": handler_not_ready,
            }
        )
        assert h.has_available_plan(DummyDef()) is False
