def addNeighboringConcepts(conceptIdsList: list[int], edgeRows: list[dict], conceptIdsToConceptNames: dict):
    conceptIdsList = set(conceptIdsList)
    for row in edgeRows: #search over all edges to find neighboring ids to search.
        if int(row["sourceconceptid"]) in conceptIdsList:
            conceptIdsList.add(row["targetconceptid"])
        elif int(row["targetconceptid"]) in conceptIdsList:
            conceptIdsList.add(row["sourceconceptid"])
    return [conceptIdsToConceptNames[int(cId)] for cId in conceptIdsList]