# Generate a corrected APC dataset from scratch using the exact noun list file.
# This will create 192 APC examples (2 per combination) with better-formed German APCs and sentences,
# plus 50 no-APC examples

import ast, random, re, pandas as pd
from pathlib import Path
random.seed(2025)

# Load noun list from uploaded file
with open("notebook/GER-nounlist-plain.txt", "r", encoding="utf-8") as f:
    content = f.read().strip()
try:
    parsed = ast.literal_eval(content)
    nouns_extracted = [str(x).strip() for x in parsed if str(x).strip()]
except Exception:
    parts = [p.strip().strip("'\"") for p in content.split(",") if p.strip()]
    nouns_extracted = parts

# Clean and deduplicate while preserving original casing where present
nouns_extracted = [n for n in nouns_extracted if n]
# Some entries have trailing spaces; strip but preserve case quirks
nouns_extracted = list(dict.fromkeys([n.strip() for n in nouns_extracted]))  # preserve order-ish
print(f"Loaded {len(nouns_extracted)} nouns from file. Sample: {nouns_extracted[:10]}")

# New nouns pool for Nnew
new_nouns = [
    "Verletzter", "Geflüchteter", "Verliebter", "Teilnehmender", "Schreibender",
    "Suchender", "Angeklagter", "Zuschauender", "Träumender", "Verzweifelter",
    "Freigeist", "Zauderer", "Musikliebhaber", "Katzenfreund", "Reisender",
    "Sprachkundige", "Hilfesuchende", "Nachdenklicher", "Aufgeschlossene",
    "Lesebegeisterter", "Neugieriger", "Umweltaktivist", "Datenanalyst", "App-Entwickler"
]

# helper functions for inflection (heuristic)
def is_nominalized_adjective(lemma):
    # consider entries that start with lowercase as nominalized adjectives (e.g., 'deutsch','arm','neu')
    return lemma and lemma[0].islower()

def cap(noun):
    return noun[0].upper() + noun[1:] if noun else noun

def pluralize(noun):
    # crude heuristics
    if not noun: return noun
    if noun.endswith(("chen","lein")):
        return noun
    if noun.endswith(("el","er","en")):
        return noun
    if noun.endswith("e"):
        return noun + "n"
    if noun.endswith("in"):
        return noun + "nen"
    # for capitalized nominalized adjectives like 'Deutsch' we prefer 'Deutschen'
    return noun + "en"

def singular_nominal_adj(noun):
    # e.g., 'deutsch' -> 'Deutscher' (masculine default)
    root = cap(noun)
    return root + "er"

def inflect_adj(stem, number):
    stem = stem.rstrip("e")
    return (stem + "en") if number=="pl" else (stem + "er")

# pronouns by case
pron_nom = {"1":"ich","2":"du","2Hon":"Sie","pl1":"wir","pl2":"ihr"}
pron_acc = {"1":"mich","2":"dich","2Hon":"Sie","pl1":"uns","pl2":"euch"}
pron_prep = {"1":"mir","2":"dir","2Hon":"Ihnen","pl1":"uns","pl2":"euch"}

constructions = ["subj","obj","PP","external"]
persons = ["1","2","2Hon"]
numbers = ["sg","pl"]
mods = ["nomod","adj","postnom","adj+postnom"]

# Build combos and plan for 2 examples each
combos = [(c,p,n,m) for c in constructions for p in persons for n in numbers for m in mods]
total_apc = len(combos)*2  # 192
target_nin = int(round(0.60*total_apc))
target_nnew = total_apc - target_nin

# Sample nouns
if target_nin <= len(nouns_extracted):
    nin_pool = random.sample(nouns_extracted, k=target_nin)
else:
    nin_pool = [random.choice(nouns_extracted) for _ in range(target_nin)]
nnew_pool = [random.choice(new_nouns) for _ in range(target_nnew)]
random.shuffle(nin_pool); random.shuffle(nnew_pool)
nin_idx = 0; nnew_idx = 0

rows = []

adj_stems = ["arm","klug","müde","glücklich","freundlich","tapfer","naiv","schlau","erschöpft","verwegen"]

for c,p,n,m in combos:
    for ex_i in range(2):
        # pick noun source to meet quotas
        remaining_nin = target_nin - nin_idx
        remaining_nnew = target_nnew - nnew_idx
        remaining_total = (total_apc) - (nin_idx + nnew_idx)
        if remaining_total>0:
            pick_nin = False
            if remaining_nin<=0:
                pick_nin=False
            elif remaining_nnew<=0:
                pick_nin=True
            else:
                pick_nin = random.random() < (remaining_nin/remaining_total)
        else:
            pick_nin = True
        if pick_nin:
            noun = nin_pool[nin_idx]
            noun_source = "NinDS"
            nin_idx += 1
        else:
            noun = nnew_pool[nnew_idx]
            noun_source = "Nnew"
            nnew_idx += 1
        # determine pronoun form depending on construction and number
        if c=="PP":
            if n=="pl":
                pron = pron_prep["pl1"] if p=="1" else (pron_prep["pl2"] if p=="2" else pron_prep["2Hon"])
            else:
                pron = pron_prep[p]
        elif c=="obj":
            if n=="pl":
                pron = pron_acc["pl1"] if p=="1" else (pron_acc["pl2"] if p=="2" else pron_acc["2Hon"])
            else:
                pron = pron_acc[p]
        else:
            if n=="pl":
                pron = pron_nom["pl1"] if p=="1" else (pron_nom["pl2"] if p=="2" else pron_nom["2Hon"])
            else:
                pron = pron_nom[p]
        # build noun form
        if is_nominalized_adjective(noun):
            if n=="pl":
                noun_form = cap(noun) + "en"
            else:
                noun_form = singular_nominal_adj(noun)
        else:
            if n=="pl":
                noun_form = pluralize(noun)
            else:
                # ensure capitalized noun
                noun_form = cap(noun) if noun[0].islower() else noun
        # modifiers handling
        post = ""
        if m in ("postnom","adj+postnom"):
            post = random.choice(["aus Berlin","aus der Vorstadt","vom Land","aus Rostock","aus Biesental","des Lichts","aus der Nachbarschaft"])
        adj_word = None
        if m in ("adj","adj+postnom"):
            stem = random.choice(adj_stems)
            adj_word = inflect_adj(stem, n)
        # Assemble APC string appropriately
        # For PP, APC should represent the complement but without preposition in APC field (user earlier had APC including pronoun and modifiers)
        components = [pron]
        if adj_word:
            components.append(adj_word)
        components.append(noun_form)
        if post:
            components.append(post)
        apc = " ".join(components)
        # Build sentence respecting construction, grammatical agreement and naturalness
        if c=="subj":
            # verb agreement
            verb = "gingen" if n=="pl" else "ging"
            sentence = f"{apc} {verb} gestern spazieren."
        elif c=="obj":
            sentence = f"Die Jury lobte {apc} in ihrer Ansprache."
        elif c=="PP":
            prep = random.choice(["mit","ohne","gegen","für","bei"])
            sentence = f"{prep} {apc} wollte niemand das Risiko eingehen."
        else: # external
            # exclamation/apposition style
            sentence = f"{apc}, so etwas hatte niemand erwartet."
        condition = f"apc-{c}-{p}-{n}-{m}-{noun_source}"
        rows.append({"example": sentence, "APC": apc, "instance": 1, "condition": condition})

# Add 50 no-APC examples (some with adjacent pronoun+noun but not APC)
no_apc = [
    "Ich wunderte mich, warum man uns Kinder nannte.",
    "Du Lehrer, kannst du mir bitte helfen?",
    "Wir trafen den alten Mann im Park.",
    "Ich fragte mich, wer diese Frau wohl ist.",
    "Er sprach mit mir und meinem Bruder.",
    "Sie erzählte von uns Freunden aus der Schulzeit.",
    "Ich erinnerte mich an den Tag mit euch.",
    "Ihr Kinder wart früher viel draußen.",
    "Wir hörten die Nachbarn laut reden.",
    "Ich sah dich, als du das Buch hobst.",
    "Sie rief uns alle zum Abendessen.",
    "Er dachte an euch Freunde aus dem Verein.",
    "Ich fragte mich, was ihr wohl denkt.",
    "Wir hörten den Lehrer sprechen.",
    "Ich traf euch auf dem Markt.",
    "Du sahst uns gestern im Theater.",
    "Ich kenne dich, aber nicht deine Schwester.",
    "Er schrieb über uns beide in seinem Tagebuch.",
    "Ihr habt uns im Garten gesehen.",
    "Ich dachte an ihn und seine Familie.",
    "Wir redeten über euch Schüler von damals.",
    "Ich wollte mit dir und den anderen sprechen.",
    "Sie sprach von uns als Kinder.",
    "Er kannte dich nur flüchtig.",
    "Ich hörte dich singen.",
    "Wir sahen sie mit ihren Freunden.",
    "Ich erkannte euch am Lachen.",
    "Ihr habt uns früher geholfen.",
    "Wir wart gemeinsam unterwegs.",
    "Ich sah euch gestern Abend.",
    "Er grüßte mich freundlich.",
    "Ich weiß, dass ihr damals geholfen habt.",
    "Wir erinnern uns an dich.",
    "Ihr wart müde nach dem Spiel.",
    "Ich hörte euch laut lachen.",
    "Er sprach über uns alle.",
    "Ich dachte an euch Menschen.",
    "Wir trafen sie zufällig im Zug.",
    "Ihr hörtet uns wohl nicht.",
    "Ich kannte dich kaum.",
    "Wir begegneten euch auf der Straße.",
    "Ich wusste, dass ihr kommen würdet.",
    "Er lobte uns für unsere Arbeit.",
    "Ich erzählte euch eine Geschichte.",
    "Ihr erinnert euch sicher.",
    "Ich sah ihn und sie tanzen.",
    "Wir halfen euch beim Umzug.",
    "Ich fragte euch nach dem Weg.",
    "Er sprach mit dir über das Projekt.",
    "Ich bemerkte euch erst spät.", #adding 50 examples from earlier generation for some choice
    "Ich wunderte mich, warum man uns Kinder nannte.",
    "Er sagte uns, Kinder sollten hier nicht spielen.",
    "Sie hob den Arm, die Lehrer riefen zur Ruhe.",
    "Man nannte sie früh die fleißigen Bäcker in der Stadt.",
    "Er meinte, wir Leute wüssten es besser.",
    "Sie trafen uns nach der Arbeit am Bahnhof.",
    "Ich sah dich gestern mit dem Bäcker im Park.",
    "Uns, die Nachbarn, störte der Lärm nur wenig.",
    "Es wunderte mich, dass man die Studenten so lobte.",
    "Man hielt ihn für einen klugen Mann, aber er blieb still.",
    "Der Richter sprach lange, die Zuhörer wurden unruhig.",
    "Die Firma stellte neue Entwickler ein, die Arbeit begann sofort.",
    "Er rief die Jungen zu sich, doch sie kamen nicht.",
    "Die alte Frau nebenan grüßte freundlich.",
    "Wir hörten von den Erfahrungen anderer Reisender.",
    "Sie nannte ihn einen alten Freund, aber das stimmte nicht.",
    "Der Lehrer erklärte die Aufgabe, die Schüler hörten zu.",
    "Sie sahen uns mit Misstrauen, aber wir blieben ruhig.",
    "Wir machten den Eindruck, wirklich müde zu sein.",
    "Du hast mich und meinen Bruder im Supermarkt gesehen.",
    "Der Bäcker und seine Frau backen jeden Morgen Brot.",
    "Er fragte uns, ob wir den Wagen gesehen hätten.",
    "Das Team und die Trainer diskutierten lange über Taktik.",
    "Man nannte sie die mutigen Frauen der Stadt.",
    "Ich dachte, du hättest den Brief an die Nachbarin geschickt.",
    "Die Journalisten berichteten, aber sie waren sich nicht einig.",
    "Sie stellten Fragen, die dem Kandidaten schwer fielen.",
    "Die Kinder aus der Klasse sangen ein Lied.",
    "Er erklärte, wir könnten später darüber sprechen.",
    "Sie riefen die Helfer zu einem Treffen am Morgen.",
    "Man erwartete uns, doch wir kamen zu spät.",
    "Die Touristen zeigten uns ihre Karten, wir halfen gern.",
    "Er lobte seine Mutter und seine Schwester öffentlich.",
    "Sie sah ihn an, der Mann wirkte sehr beschäftigt.",
    "Ich traf den Bäcker, er erzählte von neuen Rezepten.",
    "Der Trainer motivierte die Spieler vor dem Spiel.",
    "Ich sah die Studentin im Lesesaal arbeiten.",
    "Der Busfahrer grüßte seine Kollegen am Morgen.",
    "Er erklärte der Jury seine Idee sehr klar.",
    "Sie stellten den neuen Kollege vor der Abteilung vor.",
    "Ich nannte den Film gut, aber nicht perfekt.",
    "Sie sprachen über uns, doch wir hörten nicht zu.",
    "Der Fahrer der Bahn kontrollierte die Fahrkarten.",
    "Die Nachbarin bringt oft Kuchen vorbei.",
    "Er sagte: ‚Ihr, Bäcker, solltet pünktlich sein.‘",
    "Die Kinder waren laut, die Eltern schämten sich.",
    "Wir beobachteten die Gruppe der Forscher am Ufer.",
    "Der Verkäufer und der Kunde unterhielten sich lange."
]
for s in no_apc[:50]:
    rows.append({"example": s, "APC": "", "instance": 0, "condition": "noapc"})

# Save
out = Path("data/tests/APCtest-GER-2.csv")
pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
out, len(rows)
