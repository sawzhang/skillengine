"""Tests for image/vision content types in models."""

from __future__ import annotations

from skillengine.models import ImageContent, MessageContent, TextContent


class TestTextContent:
    """Tests for TextContent dataclass."""

    def test_default_values(self) -> None:
        """TextContent should have correct defaults."""
        tc = TextContent()
        assert tc.type == "text"
        assert tc.text == ""

    def test_custom_text(self) -> None:
        """TextContent should accept custom text."""
        tc = TextContent(text="hello world")
        assert tc.type == "text"
        assert tc.text == "hello world"

    def test_custom_type_and_text(self) -> None:
        """TextContent fields can be overridden."""
        tc = TextContent(type="custom", text="payload")
        assert tc.type == "custom"
        assert tc.text == "payload"


class TestImageContent:
    """Tests for ImageContent dataclass."""

    def test_default_values(self) -> None:
        """ImageContent should have correct defaults."""
        ic = ImageContent()
        assert ic.type == "image"
        assert ic.data == ""
        assert ic.mime_type == "image/png"

    def test_custom_data(self) -> None:
        """ImageContent should accept custom base64 data."""
        ic = ImageContent(data="aW1hZ2VkYXRh")
        assert ic.data == "aW1hZ2VkYXRh"
        assert ic.mime_type == "image/png"

    def test_custom_mime_type(self) -> None:
        """ImageContent should accept custom mime_type."""
        ic = ImageContent(data="abc123", mime_type="image/jpeg")
        assert ic.type == "image"
        assert ic.data == "abc123"
        assert ic.mime_type == "image/jpeg"

    def test_webp_mime_type(self) -> None:
        """ImageContent should accept webp mime_type."""
        ic = ImageContent(mime_type="image/webp")
        assert ic.mime_type == "image/webp"


class TestMessageContent:
    """Tests for the MessageContent union type."""

    def test_plain_string(self) -> None:
        """MessageContent can be a plain string."""
        content: MessageContent = "hello"
        assert isinstance(content, str)
        assert content == "hello"

    def test_list_of_text_content(self) -> None:
        """MessageContent can be a list of TextContent."""
        content: MessageContent = [
            TextContent(text="first"),
            TextContent(text="second"),
        ]
        assert isinstance(content, list)
        assert len(content) == 2
        assert all(isinstance(item, TextContent) for item in content)
        assert content[0].text == "first"
        assert content[1].text == "second"

    def test_list_of_image_content(self) -> None:
        """MessageContent can be a list of ImageContent."""
        content: MessageContent = [
            ImageContent(data="img1", mime_type="image/png"),
            ImageContent(data="img2", mime_type="image/jpeg"),
        ]
        assert isinstance(content, list)
        assert len(content) == 2
        assert all(isinstance(item, ImageContent) for item in content)
        assert content[0].data == "img1"
        assert content[1].mime_type == "image/jpeg"

    def test_mixed_list(self) -> None:
        """MessageContent can be a mixed list of TextContent and ImageContent."""
        content: MessageContent = [
            TextContent(text="Look at this image:"),
            ImageContent(data="aW1n", mime_type="image/png"),
            TextContent(text="What do you see?"),
        ]
        assert isinstance(content, list)
        assert len(content) == 3
        assert isinstance(content[0], TextContent)
        assert isinstance(content[1], ImageContent)
        assert isinstance(content[2], TextContent)
        assert content[0].text == "Look at this image:"
        assert content[1].data == "aW1n"
        assert content[2].text == "What do you see?"

    def test_empty_list(self) -> None:
        """MessageContent can be an empty list."""
        content: MessageContent = []
        assert isinstance(content, list)
        assert len(content) == 0
