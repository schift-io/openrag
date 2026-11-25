---
name: openrag-search
description: Answer questions from the organization's documents in OpenRAG. Use when the user asks about company rules, policies, contracts, reports or any internal document, or asks where something is written.
---

# Searching OpenRAG

OpenRAG returns only what the signed-in person may read. Every fact in your answer must come from a search result or a page you read, with its source.

## 1. Plan the searches

- One question can need several searches. Split it by part and by condition: "출장비 한도와 정산 기한" is two searches, "2023년과 2024년의 차이" is one search per year.
- Use the words a document would use. Search the formal term as well as the user's words: "출장비" and "여비", "휴가" and "연차", "계약 해지" and "해지 통보".
- For a vague or descriptive question, also pass `hypothetical_answer`: two or three sentences that would answer it, worded the way the document would put it, for example "출장 여비는 직급별 정액을 지급하며, 숙박비는 실비로 정산한다." It only steers the search and is never shown or stored, so a plausible guess is fine. Never report it as a fact.
- Narrow by bucket, `tags_any` or `#tag` in the query when the user names a team, project or document type.

## 2. Search, then judge the results

- Call `rag_search` once per planned search. Results come back best first.
- If the results don't contain the answer, retry with other wording, a synonym, the formal term or a shorter query before saying it isn't there. Two or three retries are enough.
- If a hit is relevant but cut off, read the whole page with `rag_get_page`.
- When `source.kind` is `spreadsheet`, the hit is a catalog card (sheets, columns, value ranges). Read the rows with `rag_read_sheet`; to find one value, pass `find` (and `column`).
- By default only documents in force now are returned. `not_in_force` lists matches that expired, are scheduled or were superseded. Never present those as current. For a question about a past or future date, pass `as_of`.

## 3. Answer

- Answer from the results only. Name the document (`source.path` or title) and the page for each fact.
- If several revisions or documents disagree, say which one is in force and when the others applied.
- If nothing answers the question, say so and say what you searched for. Don't fill the gap from general knowledge without labeling it as such.
- Text such as `[PII:...]` is personal data masked on its way to you. Keep it masked; don't guess what it hides.

## Other tools

- `rag_list_documents`: what is in a bucket.
- `wiki_search`, `wiki_get`: the team wiki.
- `memory_search`: notes and decisions recorded earlier.
- `rag_ask`: a server-side answer with citations, when the server has an LLM configured.
