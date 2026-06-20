from chunking.schemas.models import ExtractedClinicalData

class MedicalNormalizer:
    def normalize(self, data: ExtractedClinicalData) -> ExtractedClinicalData:
        for chunk in data.chunks:
            for entity in chunk.entities:
                entity.normalized_name = entity.name.lower().strip()
        return data
