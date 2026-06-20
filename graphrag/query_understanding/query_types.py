from enum import Enum


class QueryType(str, Enum):
    SYMPTOM_QUERY      = "symptom_query"
    DRUG_INTERACTION   = "drug_interaction"
    DIAGNOSIS          = "diagnosis"
    GUIDELINE          = "guideline"
    LAB_INTERPRETATION = "lab_interpretation"
    PROGNOSIS          = "prognosis"
    OUT_OF_CONTEXT     = "out_of_context"
    UNKNOWN            = "unknown"
