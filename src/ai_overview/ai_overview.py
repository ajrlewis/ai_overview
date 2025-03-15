"""
The code generates a comprehensive summary or overview based on an input query,
leveraging query expansion, search, text extraction to produce a  rich output
containing relevant information and links.
"""

import datetime
import json
import re
import sys
import time
from typing import Optional

from aikit.chat import prompt_templates, tools
from aikit.client import hugging_face_client, open_ai_client
from loguru import logger
import webkit

# Configure LLM

# client = hugging_face_client.get_client()
# model = "mistralai/Mistral-7B-Instruct-v0.3"
client = open_ai_client.get_client()
model = "gpt-3.5-turbo"

# Module variables

# Higher values can increase noise and drown signal, whilst lower
# values may result in too weak a signal.
number_of_additional_queries = 3  # The number of additional queries to generate
number_of_search_results = 5  # The number of search results per query

# LLM Prompts

QUERY_SANITIZATION_TEMPLATE = """
Sanitize the following search query:

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

OVERVIEW_TEMPLATE = """
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


# def recognise_named_entities(text: str) -> list[dict]:
#     logger.debug(f"{text = }")
#     named_entities = tools.named_entity_recognition(
#         client=client, model=model, text=text
#     )
#     logger.debug(f"{named_entities = }")
#     return named_entities
def recognise_named_entities(text: str) -> list[dict]:
    """
    Recognises named entities in the provided text.

    This method uses a named entity recognition model to identify and extract named entities
    from the provided text. The extracted entities are returned as a list of dictionaries,
    where each dictionary contains information about the entity.

    Args:
        text (str): The text to extract named entities from.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains information about a named entity.

    Raises:
        Exception: If an error occurs during named entity recognition.
    """
    logger.debug(f"Recognising named entities in text: {len(text)} characters")
    try:
        # Perform named entity recognition on the text
        named_entities = tools.named_entity_recognition(
            client=client, model=model, text=text
        )
        logger.debug(f"Extracted {len(named_entities)} named entities")
        return named_entities
    except Exception as e:
        logger.error(f"Error recognising named entities: {e}")


def generate_overview(text: str, queries: list[str]) -> str:
    """
    Generates an overview based on the provided text and queries.

    This method uses a language model to generate an overview by asking a question
    that includes the provided text and queries. The question is formatted using the
    OVERVIEW_TEMPLATE template.

    Args:
        text (str): The text to include in the overview.
        queries (list[str]): The queries to include in the overview.

    Returns:
        str: The generated overview.

    Raises:
        Exception: If an error occurs during the generation of the overview.
    """
    logger.debug(
        f"Generating overview for text: {len(text)} characters, queries: {len(queries)}"
    )
    try:
        # Format the question using the OVERVIEW_TEMPLATE template
        question = OVERVIEW_TEMPLATE.format(text=text, queries=queries)
        logger.debug(f"Formatted question: {len(question)} characters")

        # Ask the language model to generate an overview
        overview = tools.ask(client=client, model=model, question=question)
        logger.debug(f"Generated overview: {len(overview)} characters")

        return overview
    except Exception as e:
        logger.error(f"Error generating overview: {e}")
        # Consider re-raising the exception or handling it in a way that makes sense for your application


# Search Result Methods.


def remove_duplicates_and_unused_keys(search_results: list[dict]) -> list[dict]:
    """
    Removes duplicate search results with the same link and removes unused keys.

    This method first removes the 'body' key from each search result, as it is not needed.
    It then removes any duplicate search results by converting the dictionaries to tuples,
    adding them to a set (which automatically removes duplicates), and then converting them back to dictionaries.

    Args:
        search_results (list[dict]): A list of dictionaries, where each dictionary represents a search result.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a unique search result with only the necessary keys.

    Raises:
        Exception: If an error occurs during the processing of the search results.
    """
    logger.debug(
        f"Removing duplicates and unused keys from {len(search_results)} search results"
    )
    try:
        # Remove the 'body' key from each search result
        search_results = [
            {k: v for k, v in r.items() if k != "body"} for r in search_results
        ]
        logger.debug(f"Removed 'body' key from search results")

        # Remove any duplicate search results
        unique_search_results = set(tuple(sorted(r.items())) for r in search_results)
        logger.debug(
            f"Removed {len(search_results) - len(unique_search_results)} duplicate search results"
        )

        # Convert the set of tuples back to a list of dictionaries
        unique_search_results = [dict(t) for t in unique_search_results]
        logger.debug(f"Processed search results: {len(unique_search_results)}")
        return unique_search_results
    except Exception as e:
        logger.error(f"Error removing duplicates and unused keys: {e}")


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


def extract_text_and_links(search_results: list[dict]) -> tuple[str, list[str]]:
    """
    Extracts the text and links from a list of search results.

    This method iterates over each search result, extracts the title, snippet, and link,
    and combines the title and snippet into a single text string. It also adds the link to a list of links.
    Finally, it joins the text strings into a single string and removes any duplicate links.

    Args:
        search_results (list[dict]): A list of dictionaries, where each dictionary represents a search result.

    Returns:
        tuple[str, list[str]]: A tuple containing the extracted text and a list of unique links.

    Raises:
        Exception: If an error occurs during the processing of the search results.
    """
    logger.debug(f"Extracting text and links from {len(search_results)} search results")
    try:
        texts = []
        links = []
        for search_result in search_results:
            # Extract the title, snippet, and link from the search result
            title = search_result.get("title", "").strip()
            snippet = search_result.get("snippet", "").strip()
            href = search_result.get("href", "").strip()

            # Add the link to the list of links
            links.append(href)

            # Combine the title and snippet into a single text string
            text = f"{title}: {snippet}"
            texts.append(text)

        # Join the text strings into a single string
        text = "\n".join(texts)
        logger.debug(f"Extracted text: {len(text)} characters")

        # Remove any duplicate links
        links = list(set(links))
        logger.debug(f"Extracted {len(links)} unique links")

        return text, links
    except Exception as e:
        logger.error(f"Error extracting text and links: {e}")


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
    """
    Performs a detailed search using AI.

    Args:
        original_query (str): The initial search query.

    Returns:
        dict: A dictionary containing the search results, including an overview, links, named entities, and the time searched.
    """

    if not original_query:
        logger.error("Initial search query not supplied!")
        return

    # Log initial search query
    logger.info("Initial search query:")
    logger.info(f"    {original_query}")

    # Sanitize initial search query
    sanitized_query = sanitize_initial_query(original_query)
    logger.info("Sanitized original search query:")
    logger.info(f"    {sanitized_query}")

    # Generate additional search queries
    additional_queries = expand_query(sanitized_query)
    logger.info(
        f"Created {len(additional_queries)} additional queries to gain a broader insight:"
    )
    for index, query in enumerate(additional_queries):
        logger.info(f"    {index + 1}. {query}")

    # Combine initial query with additional queries
    queries = [sanitized_query] + additional_queries

    # Search each query and measure time
    start_time = time.time()
    search_results = perform_searches(queries)
    end_time = time.time()
    logger.info(
        f"Found {len(search_results)} related search results in {round(end_time - start_time, 2)} seconds."
    )

    # Remove duplicate search results and unused keys
    unique_search_results = remove_duplicates_and_unused_keys(search_results)
    logger.info(
        f"Removed {len(search_results) - len(unique_search_results)} duplicate search results."
    )
    logger.info(f"Analyzing the following {len(unique_search_results)} search results:")
    for index, result in enumerate(unique_search_results):
        title = result.get("title")
        logger.info(f"    {index + 1}. {title}")

    # Extract text and links from search results
    text, links = extract_text_and_links(unique_search_results)

    # Generate overview
    overview = generate_overview(text, queries)
    logger.info("Generated overview:")
    logger.info(f"{overview}")

    # Perform named entity recognition
    named_entities = recognise_named_entities(overview)
    # logger.info("Named entities:")
    # for index, named_entity in enumerate(named_entities):
    #     entity_category = named_entity["category"]
    #     entity_type = named_entity["type"]
    #     entity_confidence = named_entity["confidence"]
    #     entity_text = named_entity["text"]
    #     logger.info(
    #         f"    {index + 1}. {entity_category}: {entity_type} [{entity_confidence}]"
    #     )
    #     logger.info(f"    {entity_text}")

    # Construct data to return
    data = {
        "overview": overview,
        "links": links,
        "named_entities": named_entities,
        "searched_on": datetime.datetime.now(),
    }
    logger.debug(f"{data = }")

    # Log duration
    duration = time.time() - start_time
    print()
    logger.info(f"Completed AI search in {round(duration, 2)} seconds!")

    return data


def main(query: str):
    data = ai_search(query)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:])
    main(query)
