from __future__ import annotations

from dataclasses import dataclass

from modules.story_splitter import StoryPart


@dataclass
class PartMetadata:
    part_index: int
    title: str
    caption: str
    hashtags: list[str]


def generate_part_metadata(base_title: str, topic: str, part: StoryPart) -> PartMetadata:
    hashtags = [
        "#shorts",
        "#storytime",
        "#viral",
        "#fyp",
        f"#{topic.lower().replace(' ', '')[:20]}",
    ]
    title = f"{base_title} | {part.part_title}"
    caption = (
        f"{part.part_title}. {part.cliffhanger} "
        f"Follow for the next chapter."
    )
    return PartMetadata(
        part_index=part.index,
        title=title,
        caption=caption,
        hashtags=hashtags,
    )
