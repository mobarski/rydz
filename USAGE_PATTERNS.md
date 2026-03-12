# Usage Patterns

## When Rydz fits

Use Rydz when an agent must make the same small decision over hundreds or thousands of noisy texts or documents.

Especially useful for:

- document triage
- page triage inside large PDFs
- evidence detection
- entity relationship scoring inside long documents
- duplicate detection
- same-event matching
- multi-stage filtering pipelines
- routing uncertain cases to humans

If a task can be phrased as `YES/NO` or a small fixed label set, Rydz probably fits.

## Common agent patterns

### Coarse filter
First pass: is this even relevant?

### Fine filter
Second pass: is it specifically about the target topic, entity, route, jurisdiction, etc.?

### Evidence detector
Does this text contain signals typical of the thing we are looking for?

### Repeated entity-query scoring
Keep one long document fixed and ask many short questions about entities, places, codes, or relationships inside it.

### Pairwise duplicate check
Do these two texts refer to the same event, booking, trip, case, person, or document?

### Human review gate
Use probability thresholds to avoid fake certainty:

- `p >= 0.85` -> accept
- `p <= 0.15` -> reject
- otherwise -> manual review

## Why this works well for agents

Agents often think in terms of search, retrieval, filtering, deduplication, and review.
Many of those steps are really just repeated classification with uncertainty handling.

Rydz is especially useful when the agent should escalate uncertain cases instead of pretending to know.

It is also a strong fit when one long document is paired with many short questions appended at the end.
That pattern is cache-friendly on hosted APIs and often much faster on local models.

## Anti-patterns / failure modes

Rydz is usually a bad fit when:

- the task is open-ended generation rather than classification
- the answer space is large, fuzzy, or keeps changing
- you need exact extraction spans, not a score or a small label
- the model is unlikely to put the answer label first
- the relevant evidence is buried in a huge document and you did not chunk or prefilter it
- one label is much safer to over-predict than under-predict, but no thresholding or manual review is used

Common ways agents fail with Rydz:

- asking vague questions such as `is this important?` instead of operational ones
- using labels that are semantically overlapping, such as `RELATED` vs `PARTLY_RELATED`
- using labels that may split into multiple tokens or be awkward for the model to emit
- trusting a raw score without checking calibration on a small sample
- skipping the uncertain middle band and forcing every case into accept or reject
- asking entity-relation questions without first generating good candidate pairs

Good fixes:

- rewrite the task as `YES/NO` or a small crisp label set
- make the labels easy to emit and easy to separate conceptually
- chunk or prefilter long documents before scoring
- use thresholds and manual review for borderline scores
- test prompts on a small labeled sample before large-scale runs

## Bad prompt -> better prompt

Bad:
`Is this document important?`

Better:
`Does this page contain flight reservation, boarding pass, or itinerary information worth keeping for further review?`

Bad:
`Is this about Poland?`

Better:
`Is this document about a flight to Poland, from Poland, or via Poland?`

Bad:
`Are these documents similar?`

Better:
`Do these two documents most likely refer to the same flight, trip, reservation, or boarding event?`

Bad:
`Is person X mentioned with Y?`

Better:
`Is there evidence in this document that person X is associated with travel to Y?`

Bad:
`Classify this into many categories...`

Better:
Split into stages:

1. `is this relevant at all?`
2. `what exact relation holds?`
3. `is the score high enough to accept automatically?`

## Single-text binary classification
```
Read the following text and answer the question at the end of this message.
Your answer must be either YES or NO - all caps, nothing else.
No spaces, no markup, no xml tags - only YES or NO.
<text>
	...
</text>
Does this look like a hotel reservation or something very similar?
Answer YES or NO.
```

## Two-text similarity classification
```
Read the following two texts and answer the question at the end of this message.
Your answer must be either YES or NO - all caps, nothing else.
No spaces, no markup, no xml tags - only YES or NO.
<text1>
	...
</text1>
<text2>
	...
</text2>
Are these two texts talking about the same event?
Answer YES or NO.
```

## Single-text multi-class classification
```
Read the following text and answer the question at the end of this message.
Your answer must be exactly one of:
TO_POLAND
FROM_POLAND
VIA_POLAND
NOT_POLAND
UNCLEAR

No spaces, no markup, no xml tags - only one label.
<text>
	...
</text>
How is Poland related to the flight described in this text?
Answer with exactly one label.
```

## Flight document detection
```
Read the following document and answer the question at the end of this message.
Your answer must be either YES or NO - all caps, nothing else.
No spaces, no markup, no xml tags - only YES or NO.
<document>
	...
</document>
Does this document describe a flight, air travel, a boarding pass, an airline reservation, an itinerary, or airport check-in information?
Answer YES or NO.
```

## Flight evidence detection
```
Read the following text and answer the question at the end of this message.
Your answer must be either YES or NO - all caps, nothing else.
No spaces, no markup, no xml tags - only YES or NO.
<text>
	...
</text>
Does this text contain evidence typical of a flight boarding pass, airline reservation, e-ticket, or airport itinerary?
Examples include airport codes, flight numbers, passenger name, gate, seat, departure, arrival, booking reference, or baggage details.
Answer YES or NO.
```

## Large-PDF page triage
```
Read the following page and answer the question at the end of this message.
Your answer must be either YES or NO - all caps, nothing else.
No spaces, no markup, no xml tags - only YES or NO.
<page>
	...
</page>
Is this page likely to contain flight-related information worth keeping for further review?
Answer YES or NO.
```

## Two-document duplicate detection
```
Read the following two documents and answer the question at the end of this message.
Your answer must be either YES or NO - all caps, nothing else.
No spaces, no markup, no xml tags - only YES or NO.
<document1>
	...
</document1>
<document2>
	...
</document2>
Do these two documents most likely refer to the same flight, trip, reservation, or boarding event, even if formatting or wording differs?
Answer YES or NO.
```

## Entity relationship scoring in a long document
```
Read the following document and answer the question at the end of this message.
Your answer must be either YES or NO - all caps, nothing else.
No spaces, no markup, no xml tags - only YES or NO.
<document>
	...
</document>
Is person X associated with a flight ticket, reservation, boarding pass, or itinerary to Y?
Answer YES or NO.
```

## Cache-friendly repeated queries over one long document

This pattern is useful when:

- one long document stays fixed
- many short questions vary at the end
- you want one score per `X,Y` pair
- you want to rank candidates by score and inspect the best ones first

Typical workflow:

1. run NER or other extraction to get people, cities, airport codes, organizations, etc.
2. generate many candidate pairs such as `person -> city` or `person -> airport code`
3. score each pair with Rydz against the same long document
4. rank by probability
5. review only the top results or uncertain middle band

Why this is efficient:

- the large document prefix stays the same
- only the short question suffix changes
- hosted APIs may reuse prompt cache, making repeated queries much cheaper
- local models may reuse KV cache, making repeated queries much faster

## Entity-to-destination strength classification
```
Read the following document and answer the question at the end of this message.
Your answer must be exactly one of:
STRONG
WEAK
NONE
UNCLEAR

No spaces, no markup, no xml tags - only one label.
<document>
	...
</document>
How strong is the evidence that person X is associated with travel to Y in this document?
Answer with exactly one label.
```

## Multi-stage filtering example

For large noisy archives, a useful pipeline is often:

1. `is this document about air travel at all?`
2. `is Poland involved?`
3. `is this a strong flight-document signal or only a weak mention?`
4. `is this a duplicate of something already kept?`
5. send low-confidence cases to manual review
