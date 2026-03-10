import re

# Common typos + Malayalam transliteration variations
REPLACE_MAP = {
    # fee ethra variations
    "etra": "ethra",
    "ethra": "ethra",
    "ethraa": "ethra",
    "feeethra": "fee ethra",

    # syllabus variations
    "sylabs": "syllabus",
    "syllabs": "syllabus",
    "syllbus": "syllabus",
    "silabus": "syllabus",

    # selenium variations
    "selenum": "selenium",
    "selinium": "selenium",
    "seleniom": "selenium",

    # contact variations
    "conct": "contact",
    "contct": "contact",
    "cntct": "contact",

    # location variations
    "loc": "location",
    "evide": "evide",
    "evde": "evide",
    "evede": "evide",
    "evidae": "evide",
}

def normalize(text: str) -> str:
    t = (text or "").lower().strip()

    # keep malayalam letters too
    t = re.sub(r"[^a-z0-9\u0d00-\u0d7f\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    words = t.split()
    fixed = [REPLACE_MAP.get(w, w) for w in words]
    return " ".join(fixed)

def detect_intent(text: str) -> str | None:
    t = normalize(text)

    # Greeting
    if t in ["hi", "hii", "hello", "hey", "hai", "ഹായ്", "ഹലോ"]:
        return "greeting"

    # CONTACT (Malayalam + English)
    if any(k in t for k in [
        "contact", "call", "phone", "number", "whatsapp",
        "എങ്ങനെ ബന്ധപ്പെടാം", "ബന്ധപ്പെട", "ഫോൺ", "നമ്പർ", "വാട്സാപ്പ്"
    ]):
        return "contact"

    # LOCATION
    if any(k in t for k in [
        "location", "address", "place", "where", "evide", "evde",
        "എവിടെ", "ലൊക്കേഷൻ", "അഡ്രസ്", "വിലാസം", "സ്ഥലം"
    ]):
        return "location"

    # FEES / PRICE
    if any(k in t for k in [
        "fee", "fees", "price", "cost", "ethra", "rate",
        "ഫീസ്", "ചെലവ്", "വില", "എത്ര", "ഫീസ് എത്ര"
    ]):
        return "pricing"

    # DURATION
    if any(k in t for k in [
        "duration", "months", "time", "how long",
        "ദൈർഘ്യം", "എത്ര മാസം", "മാസം", "കഴിഞ്ഞു"
    ]):
        return "duration"

    # SYLLABUS
    if any(k in t for k in [
        "syllabus", "topics", "subject",
        "സിലബസ്", "സിലബസ് എന്താ", "വിഷയം", "ടോപ്പിക്"
    ]):
        return "syllabus"

    # DEMO
    if any(k in t for k in ["demo", "ഡെമോ", "ട്രയൽ"]):
        return "demo"

    # PLACEMENT
    if any(k in t for k in ["placement", "പ്ലേസ്മെന്റ്", "ജോലി"]):
        return "placement"

    return None

def detect_course(text: str) -> str | None:
    t = normalize(text)

    # MANUAL
    if any(k in t for k in ["manual", "മാനുവൽ", "മാനുവല്"]):
        return "manual"

    # SELENIUM/AUTOMATION
    if any(k in t for k in ["selenium", "automation", "webdriver", "java", "ഓട്ടോമേഷൻ", "സെലേനിയം", "ജാവ"]):
        return "selenium"

    # COMBINED/MASTER
    if any(k in t for k in ["master", "combined", "both", "combo", "bundle", "രണ്ടും", "മാസ്റ്റർ", "കോമ്പൈൻഡ്"]):
        return "combined"

    return None