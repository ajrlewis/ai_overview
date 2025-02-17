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
number_of_additional_queries = 3  # The number of additional queries to generate
number_of_search_results = 5  # The number of search results per query


def expand_query(query: str) -> list[str]:
    logger.debug(f"{query = }")
    question = f"""
Analyze the following search query:

    {query}

Suggest {number_of_additional_queries} alternative but related queries to pass to a search engine.

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
            query, max_results=number_of_search_results
        )
        list_of_search_results.extend(search_results)
    return list_of_search_results


# def search_results_relevance(
#     queries: list[str], search_results: list[str]
# ) -> list[bool]:
#     logger.debug(f"{queries = } {search_results = }")
#     question = f"""
# Analyze the following search results:

#     {search_results}

# For each search result, assess whether its `title`, `snippet` and `href` are relevant to the following queries:

#     {queries}

# Return a list of boolean values, e.g.


#     [ true/false, true/false, ... ]
# """
#     relevances = tools.ask(
#         client=client, model=model, question=question, parse_json=True
#     )
#     logger.debug(f"{relevances = }")
#     return relevances


def search_results_relevance(query: str, search_results: list[str]) -> list[float]:
    logger.debug(f"{query = } {search_results = }")
    question = f"""
Analyze the following search results:

    {search_results}

Compute a percentage of how relevant each search result is to the following query:

    {query}

Return your answer as a list of float values, e.g.

    [ 0.0, 100.0, 20.0, ... ]
"""
    print("relevances = ", relevances)
    relevances = tools.ask(client=client, model=model, question=question)
    logger.debug(f"{relevances = }")
    print("relevances = ", relevances)
    return relevances


def generate_overview(text: str, queries: list[str]) -> str:
    logger.debug(f"{text = } {queries = }")
    question = f"""
Analyze the following text:

    {text}

Generate an informative overview using the text that answers the following queries:

    {queries}

Note.
    - If a segment of the text does not contain the required information then ignore it.
    - If a segment of the text is not applicable to the queries then ignore it.
    - If a segment of the text is duplicated then ignore it.
    - If a segment of the text does not fit in with the majority of the other text, then ignore it.
    - Do not refer to the "text" explicitly, rather the subject at hand

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
    _search_results = [
        {k: v for k, v in r.items() if k in ("snippet")} for r in search_results
    ]
    relevances = search_results_relevance(queries, _search_results)
    # Remove search results that are irrelevant to the query
    relevant_search_results = []
    # for relevance, search_result in zip(relevances, search_results):
    #     if relevance:
    #         relevant_search_results.append(search_result)
    return relevant_search_results


def _extract_text_and_links(search_results: list[dict]) -> tuple[str, list[str]]:
    texts = []
    links = []
    for search_result in search_results:
        title = search_result.get("title")
        snippet = search_result.get("snippet")
        href = search_result.get("href")
        links.append(href)
        text = f"{title} {snippet}"
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
        logger.info("Generated AI overview:")
        logger.info(overview)
    if links := data.get("links"):
        print()
        logger.info("Using the following links:")
        for index, link in enumerate(links):
            logger.info(f"{index + 1}. {link}")


def ai_search(original_query: str = "") -> Optional[dict]:
    now = time.time()
    if not original_query:
        logger.error(f"Initial search query not supplied!")
        return
    logger.info(f"Initial search query:")
    logger.info(f"    {original_query}")

    # Generate additional search queries
    additional_queries = expand_query(original_query)
    logger.info(
        f"Created {number_of_additional_queries} additional queries to gain a broader insight:"
    )
    for query in additional_queries:
        logger.info(f"    {query}")

    queries = [original_query] + additional_queries  # Add original query

    # Search each query.
    search_results = perform_searches(queries)
    logger.info(f"Found {len(search_results)} related search results.")

    # Remove unused keys
    unique_search_results = _remove_duplicates_and_unused_keys(search_results)
    logger.info(
        f"Removed {len(search_results) - len(unique_search_results)} duplicate search results."
    )

    logger.info(f"Analyzing the following {len(unique_search_results)} search results:")
    for index, result in enumerate(unique_search_results):
        logger.info(f"    {index + 1}. {result.get('title')}")

    # Remove search results that are irrelevant to the query
    # relevant_search_results = _remove_irrelevant_results(
    #     original_query, unique_search_results
    # )
    # logger.info(
    #     f"Removed {len(unique_search_results) - len(relevant_search_results)} irrelevant search results."
    # )
    relevant_search_results = unique_search_results

    # Extract relevant text and links from search results
    text, links = _extract_text_and_links(relevant_search_results)
    logger.debug(f"{text = }")
    logger.debug(f"{links = }")

    # Generate overview.
    data = _create_summary_from_text(text, queries)
    data["links"] = links

    # Named entity recognition.
    named_entities = _recognise_named_entities(data.get("overview"))
    logger.debug(f"{named_entities = }")
    data["named_entities"] = named_entities

    # Log overview generated.
    logger.debug(f"{data = }")
    _log_overview(data)

    # Log duration
    duration = time.time() - now
    print()
    logger.info(f"Completed AI search in {round(duration)} seconds!")

    return data


def main(query: str):
    data = ai_search(query)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:])
    main(query)
