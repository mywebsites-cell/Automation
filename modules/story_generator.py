from __future__ import annotations

from dataclasses import dataclass

from modules.llm_client import LLMClient


@dataclass
class GeneratedStory:
    title: str
    narrative: str
    narrator_gender: str = ""
    narrator_tone: str = ""


@dataclass
class YouTubeMetadata:
    title: str
    description: str
    tags: list[str]


class StoryGenerator:
    """Provider-agnostic story + metadata generator built on a unified LLMClient."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate_story(
        self,
        topic: str,
        requested_parts: int,
        target_total_words: int | None = None,
        generation_instructions: str = "",
    ) -> GeneratedStory:
        target_words = max(
            200,
            int(target_total_words) if target_total_words is not None else requested_parts * 145,
        )
        prompt = (
            "You are an elite short-form storyteller. "
            "Return only valid JSON with keys: title, narrative, narrator_gender, narrator_tone. "
            "No markdown, no commentary. "
            + (
                "MANDATORY ACCOUNT RULES — these override everything else including the topic direction. "
                "You MUST follow every instruction below exactly as written. "
                "Do NOT weaken, ignore, reinterpret, or contradict them for any reason. "
                f"Rules: {generation_instructions.strip()} "
                if generation_instructions.strip()
                else ""
            )
            +
            f"Topic: {topic}. "
            f"Write one connected long-form story that can be split into about {requested_parts} parts. "
            "Narrative must be immersive, dramatic, and continuity-safe across parts. "
            f"Use around {target_words} words in total. "
            "Keep each section naturally cliffhanger-friendly for 60-second narration chunks. "
            "Choose the most fitting narrator for this story: narrator_gender must be exactly "
            "\"female\" or \"male\", and narrator_tone must be one of "
            "\"deep\", \"warm\", \"bright\", \"clear\", \"calm\"."
        )
        payload = self.llm.generate_json(prompt)
        title = str(payload.get("title", "")).strip()
        narrative = str(payload.get("narrative", "")).strip()
        narrator_gender = str(payload.get("narrator_gender", "")).strip().lower()
        narrator_tone = str(payload.get("narrator_tone", "")).strip().lower()
        if not title or not narrative:
            raise ValueError("LLM returned empty title or narrative")
        return GeneratedStory(
            title=title,
            narrative=narrative,
            narrator_gender=narrator_gender,
            narrator_tone=narrator_tone,
        )

    def generate_youtube_metadata(
        self,
        story_title: str,
        topic: str,
        part_index: int,
        part_title: str,
        part_text: str,
    ) -> YouTubeMetadata:
        """Generate an attention-grabbing YouTube Shorts title, description and tags."""
        prompt = (
            "You are a social media expert for YouTube Shorts. "
            "Return only valid JSON with keys: title, description, tags. "
            "No markdown, no commentary. "
            f"Story title: {story_title}. Topic: {topic}. "
            f"This is part {part_index} titled '{part_title}'. "
            f"Script: {part_text}. "
            "Write an attention-grabbing YouTube Shorts title (under 100 characters), "
            "a short description with a hook and a call-to-action to watch the next part, "
            "and a list of 5-8 relevant hashtags as an array of strings without the '#' symbol."
        )
        payload = self.llm.generate_json(prompt)
        title = str(payload.get("title", "")).strip()
        description = str(payload.get("description", "")).strip()
        tags = [
            str(t).strip().lstrip("#")
            for t in payload.get("tags", [])
            if str(t).strip()
        ]
        if not title or not description:
            raise ValueError("LLM returned empty YouTube metadata")
        return YouTubeMetadata(title=title, description=description, tags=tags)

    def generate_thumbnail_prompt(self, story_title: str, topic: str) -> str:
        """Return a detailed image-generation prompt for a vertical story thumbnail."""
        prompt = (
            "You are a professional YouTube thumbnail designer. "
            "Describe a single striking vertical (9:16) thumbnail image for a story video, "
            "with dramatic lighting, high contrast, and a clear focal subject. "
            "Do NOT include any text or words in the image. "
            "Return only the visual description, no commentary, no quotes."
            f"\nStory title: {story_title}.\nTopic: {topic}."
        )
        return self.llm.generate_text(prompt).strip()
