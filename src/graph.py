def addNeighboringConcepts(conceptIdsList: list[int], edgeRows: list[dict], conceptIdsToConceptNames: dict):
    conceptIdsSet = set(conceptIdsList)
    for rowTuple in edgeRows: #search over all edges to find neighboring ids to search.

        #this version grabs only the prereq concepts.
        if int(rowTuple[2]) in conceptIdsList: #[2] is target concept id
            conceptIdsSet.add(rowTuple[1]) #[1] is source concept id
    
    return [conceptIdsToConceptNames[int(cId)] for cId in conceptIdsSet]