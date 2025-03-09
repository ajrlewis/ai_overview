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


# LLM Prompts

QUERY_SANITIZATION_TEMPLATE = """
Sanitize (i.e. correct any spelling mistakes, etc.) the following search query:

    {query}

Return only the sanitized search query in your response.
"""

EXPAND_QUERY_TEMPLATE = """
Generate {number_of_additional_queries} related search queries based on:

    {query}

Return a list of alternative queries as strings, for example:

    [
        "related query 1",
        "related query 2",
        ...
    ]
"""


RELEVANCE_PROMPT = """
Analyze the following search results:

    {search_results}

Compute a percentage of how relevant each search result is to the following query:

    {query}

Return your answer as a list of float values, e.g.

    [ 0.0, 100.0, 20.0, ... ]
"""

OVERVIEW_PROMPT = """
Analyze the provided information:

    {text}

Create a concise overview that addresses the following questions:

    {queries}

When generating the overview, disregard any information that:
- does not contain the required details
- is not relevant to the questions
- is duplicated
- appears inconsistent with the majority of the information

Present the overview in a straightforward manner, focusing on the subject matter.
"""

# LLM Methods


def sanitize_initial_query(query: str) -> str:
    """
    Sanitizes the initial query by formatting it into a prompt and then using a language model to generate a sanitized version.

    Args:
        query (str): The initial query to be sanitized.

    Returns:
        str: The sanitized query.

    Raises:
        Exception: If an error occurs during the sanitization process.
    """
    logger.debug(f"{query = }")
    question = QUERY_SANITIZATION_TEMPLATE.format(query=query)
    try:
        sanitized_query = tools.ask(client=client, model=model, question=question)
        logger.debug(f"{sanitized_query = }")
        return sanitized_query
    except Exception as e:
        logger.error(f"Error sanitizing query: {e}")
        raise


def expand_query(query: str) -> list[str]:
    """
    Expands a given query into a list of related queries using a language model.

    Args:
        query (str): The query to be expanded.

    Returns:
        list[str]: A list of related queries.

    Raises:
        json.JSONDecodeError: If the response from the language model is not valid JSON.
    """
    logger.debug(f"{query = }")
    question = EXPAND_QUERY_TEMPLATE.format(
        query=query, number_of_additional_queries=number_of_additional_queries
    )
    try:
        response = tools.ask(client=client, model=model, question=question)
        logger.debug(f"{response = }")
        queries = json.loads(response)
        logger.debug(f"{queries = }")
        # Ensure the response is a list of strings
        if not isinstance(queries, list) or not all(
            isinstance(q, str) for q in queries
        ):
            logger.error(f"Invalid response from language model: {response}")
            raise ValueError("Invalid response from language model")
        return queries
    except Exception as e:
        logger.error(f"Error expanding query: {e}")
        raise


def search_results_relevance(query: str, search_results: list[str]) -> list[float]:
    logger.debug(f"{query = } {search_results = }")
    question = RELEVANCE_PROMPT.format(query=query, search_results=search_results)
    relevances = tools.ask(client=client, model=model, question=question)
    print("relevances = ", relevances)
    logger.debug(f"{relevances = }")
    return relevances


def generate_overview(text: str, queries: list[str]) -> str:
    logger.debug(f"{text = } {queries = }")
    question = OVERVIEW_PROMPT.format(text=text, queries=queries)
    overview = tools.ask(client=client, model=model, question=question)
    logger.debug(f"{overview = }")
    return overview


#


def _create_summary_from_text(text: str, queries: list[str]) -> dict:
    data = {}
    overview = generate_overview(text, queries)
    data["overview"] = overview
    return data


# Search Result Methods.


def _remove_duplicates_and_unused_keys(search_results: list[dict]) -> list[dict]:
    """Removes duplicate search results with the same link.
    Keeps only body text from search result and removes any duplicate"""
    search_results = [
        {k: v for k, v in r.items() if k not in ("body")} for r in search_results
    ]
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


def perform_searches(queries: list[str]) -> list[dict]:
    """
    Performs Google searches for a list of queries and returns the combined search results.

    Args:
        queries (list[str]): A list of queries to search for.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a search result.

    Raises:
        Exception: If an error occurs during the search process.
    """
    logger.debug(f"Performing searches for {len(queries)} queries")
    list_of_search_results = []
    for query in queries:
        logger.debug(f"Searching for: {query}")
        try:
            search_results = webkit.search.google(
                query, max_results=number_of_search_results
            )
            logger.debug(f"Found {len(search_results)} results for query: {query}")
            list_of_search_results.extend(search_results)
        except Exception as e:
            logger.error(f"Error searching for query: {query}, {e}")
            pass
    logger.debug(f"Total search results: {len(list_of_search_results)}")
    return list_of_search_results


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

    # 1. Sanitize initial search query
    sanitized_original_query = sanitize_initial_query(original_query)
    logger.info(f"Sanitized original search query:")
    logger.info(f"    {sanitized_original_query}")

    # 2. Generate additional search queries
    additional_queries = expand_query(sanitized_original_query)
    logger.info(
        f"Created {number_of_additional_queries} additional queries to gain a broader insight:"
    )
    for index, query in enumerate(additional_queries):
        logger.info(f"    {index + 1}. {query}")
    queries = [sanitized_original_query] + additional_queries  # Add initial query

    # 3. Search each query
    search_results = perform_searches(queries)
    logger.info(f"Found {len(search_results)} related search results.")
    sys.exit()

    # 4. Remove duplicate search results and unused keys
    unique_search_results = _remove_duplicates_and_unused_keys(search_results)
    logger.info(
        f"Removed {len(search_results) - len(unique_search_results)} duplicate search results."
    )
    logger.info(f"Analyzing the following {len(unique_search_results)} search results:")
    for index, result in enumerate(unique_search_results):
        title = result.get("title")
        logger.info(f"    {index + 1}. {title}")

    # Remove search results that are irrelevant to the query
    # relevant_search_results = _remove_irrelevant_results(
    #     original_query, unique_search_results
    # )
    # logger.info(
    #     f"Removed {len(unique_search_results) - len(relevant_search_results)} irrelevant search results."
    # )
    relevant_search_results = unique_search_results

    # 5. Extract relevant text and links from search results
    text, links = _extract_text_and_links(relevant_search_results)

    # 6. Generate overview.
    # data = _create_summary_from_text(text, queries)
    data = _create_summary_from_text(text, sanitized_original_query)
    data["links"] = links

    # # 7. Named entity recognition.
    # named_entities = _recognise_named_entities(data.get("overview"))
    # logger.debug(f"{named_entities = }")
    # data["named_entities"] = named_entities

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
