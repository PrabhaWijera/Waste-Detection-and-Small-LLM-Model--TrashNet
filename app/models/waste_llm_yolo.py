# app/models/waste_llm.py
class WasteLLM:
    def recommend(self, text: str, region="LK-11", city="Colombo") -> str:
        text = text.lower()
        if any(w in text for w in ["plastic", "bottle", "pet"]):
            return "Recycle plastics in designated bins. Avoid burning plastics."
        elif any(w in text for w in ["paper", "magazine", "newspaper"]):
            return "Paper waste can be reused or recycled. Compost if possible."
        elif any(w in text for w in ["glass", "jar", "bottle glass"]):
            return "Glass bottles/jars go to glass recycling centers."
        elif any(w in text for w in ["metal", "tin", "aluminum"]):
            return "Metal cans/containers should be collected for recycling."
        elif any(w in text for w in ["cardboard", "carton", "box"]):
            return "Flatten cardboard boxes and recycle them."
        else:
            return "Dispose according to local guidelines."
