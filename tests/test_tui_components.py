"""Tests for TUI components: Component, Container, InputWidget, SelectList."""

from __future__ import annotations

import pytest

from skillengine.tui.component import Component
from skillengine.tui.container import Container
from skillengine.tui.input_widget import InputWidget
from skillengine.tui.keys import Key, KEY_UP, KEY_DOWN, KEY_ENTER, KEY_LEFT, KEY_RIGHT
from skillengine.tui.select_list import ListItem, SelectList


# ---------------------------------------------------------------------------
# Concrete subclass for testing the abstract Component
# ---------------------------------------------------------------------------


class StubComponent(Component):
    """Minimal concrete component for testing the base class."""

    def __init__(self, lines: list[str] | None = None) -> None:
        super().__init__()
        self._lines = lines or ["stub"]

    def render(self, width: int) -> list[str]:
        self._dirty = False
        return self._lines


# ---------------------------------------------------------------------------
# TestComponent
# ---------------------------------------------------------------------------


class TestComponent:
    """Tests for the abstract Component base class."""

    def test_initial_dirty_state(self) -> None:
        comp = StubComponent()
        assert comp.dirty is True

    def test_render_clears_dirty(self) -> None:
        comp = StubComponent()
        comp.render(80)
        assert comp.dirty is False

    def test_invalidate_marks_dirty(self) -> None:
        comp = StubComponent()
        comp.render(80)
        assert comp.dirty is False
        comp.invalidate()
        assert comp.dirty is True

    def test_initial_visible(self) -> None:
        comp = StubComponent()
        assert comp.visible is True

    def test_initial_focused(self) -> None:
        comp = StubComponent()
        assert comp.focused is False

    def test_visible_setter_marks_dirty(self) -> None:
        comp = StubComponent()
        comp.render(80)
        assert comp.dirty is False
        comp.visible = False
        assert comp.visible is False
        assert comp.dirty is True

    def test_visible_setter_same_value_does_not_mark_dirty(self) -> None:
        comp = StubComponent()
        comp.render(80)
        assert comp.dirty is False
        comp.visible = True  # same as current
        assert comp.dirty is False

    def test_focused_setter_marks_dirty(self) -> None:
        comp = StubComponent()
        comp.render(80)
        assert comp.dirty is False
        comp.focused = True
        assert comp.focused is True
        assert comp.dirty is True

    def test_focused_setter_same_value_does_not_mark_dirty(self) -> None:
        comp = StubComponent()
        comp.render(80)
        assert comp.dirty is False
        comp.focused = False  # same as current
        assert comp.dirty is False

    def test_handle_input_returns_false_by_default(self) -> None:
        comp = StubComponent()
        key = Key(name="a", char="a")
        assert comp.handle_input(key) is False

    def test_dirty_setter(self) -> None:
        comp = StubComponent()
        comp.dirty = False
        assert comp.dirty is False
        comp.dirty = True
        assert comp.dirty is True


# ---------------------------------------------------------------------------
# TestContainer
# ---------------------------------------------------------------------------


class TestContainer:
    """Tests for the Container component."""

    def test_empty_container_renders_no_lines(self) -> None:
        container = Container()
        lines = container.render(80)
        assert lines == []

    def test_render_multiple_children(self) -> None:
        child1 = StubComponent(["line1a", "line1b"])
        child2 = StubComponent(["line2a"])
        container = Container(children=[child1, child2])
        lines = container.render(80)
        assert lines == ["line1a", "line1b", "line2a"]

    def test_render_skips_invisible_children(self) -> None:
        child1 = StubComponent(["visible"])
        child2 = StubComponent(["hidden"])
        child2.visible = False
        container = Container(children=[child1, child2])
        lines = container.render(80)
        assert lines == ["visible"]

    def test_add_child(self) -> None:
        container = Container()
        child = StubComponent(["added"])
        container.add(child)
        assert len(container.children) == 1
        assert container.children[0] is child

    def test_add_invalidates(self) -> None:
        container = Container()
        container.render(80)
        container.dirty = False
        container.add(StubComponent(["new"]))
        assert container.dirty is True

    def test_remove_child(self) -> None:
        child = StubComponent(["to-remove"])
        container = Container(children=[child])
        container.remove(child)
        assert len(container.children) == 0

    def test_insert_child(self) -> None:
        child1 = StubComponent(["first"])
        child2 = StubComponent(["second"])
        container = Container(children=[child2])
        container.insert(0, child1)
        assert container.children[0] is child1
        assert container.children[1] is child2

    def test_clear_children(self) -> None:
        container = Container(children=[StubComponent(), StubComponent()])
        container.clear()
        assert len(container.children) == 0

    def test_dirty_propagates_from_children(self) -> None:
        child = StubComponent()
        container = Container(children=[child])
        container.render(80)
        container.dirty = False
        # Child was rendered so it is clean; container is clean too.
        assert container.dirty is False
        child.invalidate()
        assert container.dirty is True

    def test_handle_input_dispatches_to_focused_child(self) -> None:
        selected: list[ListItem] = []
        item = ListItem(label="Option A", value="a")
        child = SelectList(items=[item], on_select=lambda i: selected.append(i))
        child.focused = True
        container = Container(children=[child])
        # Pressing enter on the focused SelectList should fire on_select
        assert container.handle_input(KEY_ENTER) is True
        assert len(selected) == 1
        assert selected[0] is item

    def test_handle_input_returns_false_when_empty(self) -> None:
        container = Container()
        assert container.handle_input(KEY_UP) is False

    def test_focused_index_clamping(self) -> None:
        container = Container(children=[StubComponent(), StubComponent()])
        container.focused_index = 99
        assert container.focused_index == 1
        container.focused_index = -5
        assert container.focused_index == 0

    def test_focus_next(self) -> None:
        c1 = StubComponent()
        c2 = StubComponent()
        container = Container(children=[c1, c2])
        container.focused_index = 0
        container.focus_next()
        assert container.focused_index == 1

    def test_focus_prev(self) -> None:
        c1 = StubComponent()
        c2 = StubComponent()
        container = Container(children=[c1, c2])
        container.focused_index = 1
        container.focus_prev()
        assert container.focused_index == 0

    def test_invalidate_cascades_to_children(self) -> None:
        child = StubComponent()
        child.render(80)
        assert child.dirty is False
        container = Container(children=[child])
        container.invalidate()
        assert child.dirty is True


# ---------------------------------------------------------------------------
# TestInputWidget
# ---------------------------------------------------------------------------


class TestInputWidget:
    """Tests for the InputWidget single-line text input."""

    def test_initial_value_is_empty(self) -> None:
        widget = InputWidget()
        assert widget.value == ""

    def test_set_value(self) -> None:
        widget = InputWidget()
        widget.value = "hello"
        assert widget.value == "hello"

    def test_render_returns_single_line(self) -> None:
        widget = InputWidget(prompt="> ")
        lines = widget.render(80)
        assert len(lines) == 1
        assert lines[0].startswith("> ")

    def test_render_contains_prompt(self) -> None:
        widget = InputWidget(prompt="$ ")
        widget.value = "test"
        lines = widget.render(80)
        assert "$ " in lines[0]

    def test_custom_prompt(self) -> None:
        widget = InputWidget(prompt=">>> ")
        assert widget.prompt == ">>> "

    def test_prompt_setter(self) -> None:
        widget = InputWidget(prompt="> ")
        widget.prompt = "$ "
        assert widget.prompt == "$ "

    def test_character_input_when_focused(self) -> None:
        widget = InputWidget()
        widget.focused = True
        key_a = Key(name="a", char="a")
        consumed = widget.handle_input(key_a)
        assert consumed is True
        assert widget.value == "a"

    def test_character_input_ignored_when_not_focused(self) -> None:
        widget = InputWidget()
        widget.focused = False
        key_a = Key(name="a", char="a")
        consumed = widget.handle_input(key_a)
        assert consumed is False
        assert widget.value == ""

    def test_backspace_deletes_character(self) -> None:
        widget = InputWidget()
        widget.focused = True
        widget.value = "abc"
        backspace = Key(name="backspace")
        widget.handle_input(backspace)
        assert widget.value == "ab"

    def test_enter_triggers_on_submit(self) -> None:
        submitted: list[str] = []
        widget = InputWidget(on_submit=lambda text: submitted.append(text))
        widget.focused = True
        widget.value = "hello"
        widget.handle_input(KEY_ENTER)
        assert submitted == ["hello"]

    def test_enter_adds_to_history(self) -> None:
        widget = InputWidget()
        widget.focused = True
        widget.value = "cmd1"
        widget.handle_input(KEY_ENTER)
        # After enter, the history should contain the submitted value.
        assert widget._history == ["cmd1"]

    def test_history_navigation(self) -> None:
        widget = InputWidget()
        widget.focused = True
        # Submit two entries
        widget.value = "first"
        widget.handle_input(KEY_ENTER)
        widget.value = "second"
        widget.handle_input(KEY_ENTER)
        widget.value = ""
        # Navigate up through history
        widget.handle_input(KEY_UP)
        assert widget.value == "second"
        widget.handle_input(KEY_UP)
        assert widget.value == "first"
        # Navigate back down
        widget.handle_input(KEY_DOWN)
        assert widget.value == "second"

    def test_cursor_movement(self) -> None:
        widget = InputWidget()
        widget.focused = True
        widget.value = "abc"
        # Cursor is at end (3); move left
        widget.handle_input(KEY_LEFT)
        assert widget._cursor == 2
        widget.handle_input(KEY_RIGHT)
        assert widget._cursor == 3

    def test_kill_to_end_of_line(self) -> None:
        widget = InputWidget()
        widget.focused = True
        widget.value = "hello world"
        # Move cursor to position 5 (after "hello")
        widget._cursor = 5
        ctrl_k = Key(name="ctrl+k", char="k", ctrl=True)
        widget.handle_input(ctrl_k)
        assert widget.value == "hello"

    def test_on_submit_setter(self) -> None:
        widget = InputWidget()
        assert widget.on_submit is None
        callback = lambda text: None
        widget.on_submit = callback
        assert widget.on_submit is callback


# ---------------------------------------------------------------------------
# TestSelectList
# ---------------------------------------------------------------------------


class TestSelectList:
    """Tests for the SelectList arrow-key navigation widget."""

    def test_empty_list(self) -> None:
        sl = SelectList()
        assert sl.items == []
        assert sl.selected_item is None

    def test_items_property(self) -> None:
        items = [ListItem(label="A"), ListItem(label="B")]
        sl = SelectList(items=items)
        assert len(sl.items) == 2
        assert sl.items[0].label == "A"

    def test_initial_selection(self) -> None:
        items = [ListItem(label="A"), ListItem(label="B")]
        sl = SelectList(items=items)
        assert sl.selected_index == 0
        assert sl.selected_item is not None
        assert sl.selected_item.label == "A"

    def test_navigate_down(self) -> None:
        items = [ListItem(label="A"), ListItem(label="B"), ListItem(label="C")]
        sl = SelectList(items=items)
        sl.focused = True
        sl.handle_input(KEY_DOWN)
        assert sl.selected_index == 1
        assert sl.selected_item is not None
        assert sl.selected_item.label == "B"

    def test_navigate_up(self) -> None:
        items = [ListItem(label="A"), ListItem(label="B")]
        sl = SelectList(items=items)
        sl.focused = True
        sl.selected_index = 1
        sl.handle_input(KEY_UP)
        assert sl.selected_index == 0

    def test_navigate_down_at_bottom_stays(self) -> None:
        items = [ListItem(label="A"), ListItem(label="B")]
        sl = SelectList(items=items)
        sl.focused = True
        sl.selected_index = 1
        sl.handle_input(KEY_DOWN)
        assert sl.selected_index == 1  # should not go beyond last item

    def test_navigate_up_at_top_stays(self) -> None:
        items = [ListItem(label="A"), ListItem(label="B")]
        sl = SelectList(items=items)
        sl.focused = True
        sl.handle_input(KEY_UP)
        assert sl.selected_index == 0  # should not go below 0

    def test_enter_fires_on_select(self) -> None:
        selected: list[ListItem] = []
        items = [ListItem(label="X", value=42)]
        sl = SelectList(items=items, on_select=lambda i: selected.append(i))
        sl.focused = True
        sl.handle_input(KEY_ENTER)
        assert len(selected) == 1
        assert selected[0].value == 42

    def test_render_output(self) -> None:
        items = [ListItem(label="Alpha"), ListItem(label="Beta")]
        sl = SelectList(items=items)
        lines = sl.render(80)
        # Should have at least one line per item
        assert len(lines) >= 2

    def test_render_marks_clean(self) -> None:
        items = [ListItem(label="A")]
        sl = SelectList(items=items)
        assert sl.dirty is True
        sl.render(80)
        assert sl.dirty is False

    def test_items_setter_resets_state(self) -> None:
        sl = SelectList(items=[ListItem(label="A"), ListItem(label="B")])
        sl.focused = True
        sl.handle_input(KEY_DOWN)
        assert sl.selected_index == 1
        # Replace items
        sl.items = [ListItem(label="X"), ListItem(label="Y"), ListItem(label="Z")]
        assert sl.selected_index == 0
        assert len(sl.items) == 3

    def test_selected_index_clamping(self) -> None:
        items = [ListItem(label="A"), ListItem(label="B")]
        sl = SelectList(items=items)
        sl.selected_index = 99
        assert sl.selected_index == 1
        sl.selected_index = -5
        assert sl.selected_index == 0

    def test_handle_input_ignored_when_not_focused(self) -> None:
        items = [ListItem(label="A"), ListItem(label="B")]
        sl = SelectList(items=items)
        sl.focused = False
        consumed = sl.handle_input(KEY_DOWN)
        assert consumed is False
        assert sl.selected_index == 0

    def test_list_item_description(self) -> None:
        item = ListItem(label="Test", value=1, description="A test item")
        assert item.description == "A test item"

    def test_max_visible(self) -> None:
        items = [ListItem(label=f"Item {i}") for i in range(20)]
        sl = SelectList(items=items, max_visible=5)
        sl.focused = True
        lines = sl.render(80)
        # Should render at most 5 item lines + possible scroll indicators
        # The exact count depends on scroll state, but should be bounded
        assert len(lines) <= 7  # 5 items + up to 2 scroll indicators

    def test_filterable_typing(self) -> None:
        items = [
            ListItem(label="apple"),
            ListItem(label="banana"),
            ListItem(label="avocado"),
        ]
        sl = SelectList(items=items, filterable=True)
        sl.focused = True
        # Type 'a' to filter
        key_a = Key(name="a", char="a")
        sl.handle_input(key_a)
        assert sl.filter_text == "a"
        # "apple" and "avocado" contain 'a'; "banana" also contains 'a' (fuzzy)
        # but the filter should at least be applied
        visible = sl._visible_items
        assert len(visible) >= 2

    def test_on_select_setter(self) -> None:
        sl = SelectList()
        assert sl.on_select is None
        callback = lambda item: None
        sl.on_select = callback
        assert sl.on_select is callback

    def test_home_key(self) -> None:
        items = [ListItem(label=f"Item {i}") for i in range(5)]
        sl = SelectList(items=items)
        sl.focused = True
        sl.selected_index = 3
        home_key = Key(name="home")
        sl.handle_input(home_key)
        assert sl.selected_index == 0

    def test_end_key(self) -> None:
        items = [ListItem(label=f"Item {i}") for i in range(5)]
        sl = SelectList(items=items)
        sl.focused = True
        end_key = Key(name="end")
        sl.handle_input(end_key)
        assert sl.selected_index == 4
