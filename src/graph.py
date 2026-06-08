def addNeighboringConcepts(conceptIdsList: list[int], edgeRows: list[dict], conceptIdsToConceptNames: dict):
    conceptIdsSet = set(conceptIdsList)
    for row in edgeRows: #search over all edges to find neighboring ids to search.

        #this version grabs only the prereq concepts.
        if int(row["targetconceptid"]) in conceptIdsList:
            conceptIdsSet.add(row["sourceconceptid"])
        
        #this version grabs all neighbors.
        # if int(row["sourceconceptid"]) in conceptIdsList:
        #     conceptIdsSet.add(row["targetconceptid"])
        # elif int(row["targetconceptid"]) in conceptIdsList:
        #     conceptIdsSet.add(row["sourceconceptid"])
    return [conceptIdsToConceptNames[int(cId)] for cId in conceptIdsSet]