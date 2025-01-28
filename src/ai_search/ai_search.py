"""
The code generates a comprehensive summary or overview based on an input query,
leveraging query expansion, search, text extraction, and named entity recognition
to produce a rich output containing relevant information and links.
"""

import json
import re
import sys
import time
from typing import Optional

from aikit.chat import prompt_templates, tools
from aikit.client import hugging_face_client
from loguru import logger
import webkit

# Configure LLM

client = hugging_face_client.get_client()
model = "mistralai/Mistral-7B-Instruct-v0.3"

# Module variables

# Higher values can increase noise and drown signal, whilst lower
# values may result in too weak a signal.
number_of_additional_queries = 1  # The number of additional queries to generate
number_of_search_results = 2  # The number of search results per query


def expand_query(query: str) -> list[str]:
    logger.debug(f"{query = }")
    question = f"""
Analyze the following search query:

    {query}

Suggest alternative queries to pass to a search engine.

Return your answer as a list of {number_of_additional_queries} strings in JSON format, e.g.

    {{
        "queries": [
            "alternative query 1",
            "alternative query 2",
            ...
        ]
    }}
"""
    data = tools.ask(client=client, model=model, question=question, parse_json=True)
    logger.debug(f"{data = }")
    queries = data.get("queries", [])
    logger.debug(f"{queries = }")
    return queries


def perform_searches(queries: list[str]) -> list[dict]:
    list_of_search_results = []
    for query in queries:
        search_results = webkit.search.google(
            # query, max_results=number_of_search_results, sort_by="date"
            query,
            max_results=number_of_search_results,
        )
        list_of_search_results.extend(search_results)
    return list_of_search_results


def search_results_relevance(
    queries: list[str], search_results: list[str]
) -> list[str]:
    logger.debug(f"{queries = } {search_results = }")
    question = f"""
Analyze the following search results:

    {search_results}

For each search result, assess whether its `title`, `snippet` and `href` are relevant to the following queries:

    {queries}

Return a list of booleans, e.g.

    [ true/false, true/false, ... ]
"""
    question = f"{prompt_templates.JSON_FORMAT} {question}"
    data = tools.ask(client=client, model=model, question=question)
    logger.debug(f"{data = }")
    relevances = json.loads(data)
    logger.debug(f"{relevances = }")
    return relevances


def generate_overview(text: str, queries: list[str]) -> str:
    logger.debug(f"{text = } {queries = }")
    question = f"""
Analyze the following text:

    {text}

Generate an informative overview of the text.

Ensure that the overview answers the following queries:

    {queries}

Style the overview as an informative AI search engine bot. Hence, do not refer to the "text" explicitly, rather the subject at hand.

Note.
    - If a segment of the text does not contain the required information then ignore it.
    - If a segment of the text is not applicable to the queries then ignore it.
    - If a segment of the text is duplicated then ignore it.
    - If a segment of the text does not fit in with the majority of the other text, then ignore it.

Do not format your result.
"""
    overview = tools.ask(client=client, model=model, question=question)
    logger.debug(f"{overview = }")
    return overview


def _create_summary_from_text(text: str, queries: list[str]) -> dict:
    data = {}
    overview = generate_overview(text, queries)
    data["overview"] = overview
    return data


def _remove_duplicates_and_unused_keys(search_results: list[dict]) -> list[dict]:
    # Remove unused keys
    search_results = [
        {k: v for k, v in r.items() if k not in ("body")} for r in search_results
    ]
    # Remove duplicates search results
    unique_search_results = set(tuple(sorted(r.items())) for r in search_results)
    unique_search_results = [dict(t) for t in unique_search_results]
    return unique_search_results


def _remove_irrelevant_results(
    queries: list[str], search_results: list[dict]
) -> list[dict]:
    # Get whether each search result is relevant to the queries
    relevances = search_results_relevance(queries, search_results)
    # Remove search results that are irrelevant to the query
    relevant_search_results = []
    for relevance, search_result in zip(relevances, unique_search_results):
        if relevance:
            relevant_search_results.append(search_result)
    return relevant_search_results


def _extract_text_and_links(search_results: list[dict]) -> tuple[str, list[str]]:
    texts = []
    links = []
    for search_result in search_results:
        title = search_result.get("title")
        snippet = search_result.get("snippet")
        href = search_result.get("href")
        links.append(href)
        text = f"{title}: {snippet}"
        texts.append(text)
    text = "\n".join(texts)
    links = list(set(links))
    return text, links


def _recognise_named_entities(text: str) -> list[dict]:
    logger.debug(f"{text = }")
    named_entities = tools.named_entity_recognition(
        client=client, model=model, text=text
    )
    logger.debug(f"{named_entities = }")
    return named_entities


def _log_overview(data: dict):
    if overview := data.get("overview"):
        print()
        logger.info("Overview :")
        logger.info("    " + overview)
    # if named_entities := data.get("named_entities"):
    #     print()
    #     logger.info("Named Entities:")
    #     for named_entity in named_entities:
    #         for key, value in named_entity.items():
    #             logger.info("    " + key + ": " + str(value))
    #         # print()
    if links := data.get("links"):
        print()
        logger.info("Links:")
        for link in links:
            logger.info("    " + link)


def ai_search(query: str) -> dict:
    now = time.time()
    logger.info(f"{query = }")

    # Generate additional search queries
    queries = expand_query(query)
    queries.append(query)  # Add original query
    logger.info(f"{queries = }")

    # Search each query.
    search_results = perform_searches(queries)
    logger.info(f"{search_results = }")

    # Remove unused keys
    unique_search_results = _remove_duplicates_and_unused_keys(search_results)
    logger.info(f"{unique_search_results = }")

    # Remove search results that are irrelevant to the query
    # relevant_search_results = _remove_irrelevant_results(queries, unique_search_results)
    relevant_search_results = unique_search_results
    logger.info(f"{relevant_search_results = }")

    # Extract relevant text and links from search results
    text, links = _extract_text_and_links(relevant_search_results)

    # Generate overview.
    data = _create_summary_from_text(text, queries)
    data["links"] = links

    # Named entity recognition.
    named_entities = _recognise_named_entities(data.get("overview"))
    logger.info(f"{named_entities = }")
    data["named_entities"] = named_entities

    # Log overview generated.
    logger.info(f"{data = }")
    _log_overview(data)

    # Log duration
    duration = time.time() - now
    logger.info(f"Completed overview in {round(duration)} seconds!")

    return data


def main(query: str):
    data = ai_search(query)


if __name__ == "__main__":
    query = sys.argv[1:]
    main(query)
