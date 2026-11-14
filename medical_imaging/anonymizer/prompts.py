"""Prompt construction and entity seeds for synthetic clinical-note generation.

Seeds are small curated samples of real Czech data (common first/last names by
frequency, real towns/streets/PSČ). Swap in full ČSÚ name-frequency tables and a
RÚIAN address dump for production-scale generation.
"""

import random
import unicodedata
from dataclasses import dataclass

ENTITY_LABELS = [
    "NAME",
    "AGE",
    "DOB",
    "ID",
    "ADDRESS",
    "PHONE",
    "EMAIL",
    "DOCTOR",
    "HOSPITAL",
]

# Most frequent Czech first names (ČSÚ frequency order), split by gender so the
# LLM inflects correctly.
FIRST_NAMES_M = [
    "Jan", "Petr", "Pavel", "Jiří", "Josef", "Jaroslav", "Miroslav", "Martin",
    "Tomáš", "František", "Zdeněk", "Karel", "Milan", "Václav", "Michal",
    "Vladimír", "Stanislav", "Ladislav", "Lukáš", "Jakub", "David", "Roman",
    "Radek", "Vlastimil", "Rudolf", "Antonín", "Oldřich", "Bohumil", "Vilém",
    "Otakar",
]
FIRST_NAMES_F = [
    "Marie", "Jana", "Eva", "Hana", "Anna", "Věra", "Lenka", "Alena", "Petra",
    "Lucie", "Jitka", "Jaroslava", "Ludmila", "Helena", "Jarmila", "Martina",
    "Zdeňka", "Ivana", "Božena", "Miroslava", "Květa", "Libuše", "Milada",
    "Soňa", "Růžena", "Vlasta", "Drahomíra", "Blanka", "Emilie", "Anežka",
]
LAST_NAMES_M = [
    "Novák", "Svoboda", "Novotný", "Dvořák", "Černý", "Procházka", "Kučera",
    "Veselý", "Horák", "Němec", "Marek", "Pospíšil", "Pokorný", "Hájek",
    "Král", "Jelínek", "Růžička", "Beneš", "Fiala", "Sedláček", "Doležal",
    "Zeman", "Kolář", "Navrátil", "Čermák", "Urban", "Vaněk", "Blažek",
    "Kříž", "Kovář", "Kratochvíl", "Bartoš", "Vlček", "Polák", "Musil",
]
LAST_NAMES_F = [
    "Nováková", "Svobodová", "Novotná", "Dvořáková", "Černá", "Procházková",
    "Kučerová", "Veselá", "Horáková", "Němcová", "Marková", "Pospíšilová",
    "Pokorná", "Hájková", "Králová", "Jelínková", "Růžičková", "Benešová",
    "Fialová", "Sedláčková", "Doležalová", "Zemanová", "Kolářová",
    "Navrátilová", "Čermáková", "Urbanová", "Vaňková", "Blažková", "Křížová",
    "Kovářová", "Kratochvílová", "Bartošová", "Vlčková", "Poláková",
    "Musilová",
]

# Real towns + real streets + matching PSČ (South Moravia / Prague mix, matching
# the region of the source export).
ADDRESSES = [
    ("Masarykovo náměstí", "Hodonín", "695 01"),
    ("Nádražní", "Kyjov", "697 01"),
    ("Svatoborice", "Kyjov", "697 01"),
    ("Komenského", "Uherské Hradiště", "686 01"),
    ("Palackého náměstí", "Kunovice", "686 04"),
    ("Hlavní", "Veselí nad Moravou", "698 01"),
    ("Brněnská", "Břeclav", "690 02"),
    ("Zámecké náměstí", "Mikulov", "692 01"),
    ("Valtická", "Valtice", "691 42"),
    ("Dukelská", "Hustopeče", "693 01"),
    ("Špitální", "Kyjov", "697 33"),
    ("Jana Palacha", "Bzenec", "696 81"),
    ("Tyršova", "Strážnice", "696 62"),
    ("Školní", "Ratíškovice", "696 02"),
    ("Pod vinohrady", "Dolní Bojanovice", "696 17"),
    ("Čelakovského", "Hodonín", "695 01"),
    ("Lipová alej", "Hodonín", "695 01"),
    ("Za nemocnicí", "Uherský Ostroh", "687 24"),
    ("Mírové náměstí", "Vnorovy", "696 61"),
    ("Zelňárky", "Rohatec", "696 01"),
    ("Žižkova", "Kroměříž", "767 01"),
    ("Purkyněova", "Vyškov", "682 01"),
    ("Studentská", "Ivančice", "664 91"),
    ("Palackého", "Rosice", "665 01"),
    ("Olšová", "Bučovice", "685 01"),
    ("Václavské náměstí", "Praha 1", "110 00"),
    ("Karlštejnská", "Praha 5", "158 00"),
    ("Vinohradská", "Praha 3", "120 00"),
    ("Gajdošova", "Brno", "615 00"),
    ("Údolní", "Brno", "602 00"),
]

HOSPITALS = [
    "Nemocnice Kyjov, ARO",
    "Nemocnice Kyjov, interní oddělení",
    "Nemocnice Kyjov, chirurgické oddělení",
    "Nemocnice Hodonín, interní oddělení",
    "Nemocnice Hodonín, JIP",
    "Nemocnice Uherské Hradiště, interní oddělení",
    "Nemocnice Břeclav, ARO",
    "Fakultní nemocnice Brno, I. interní klinika",
    "Fakultní nemocnice u sv. Anny v Brně",
    "Nemocnice Vyškov, interní oddělení",
]

DIAGNOSES = [
    "J18.9 bronchopneumonie",
    "I46.0 srdeční zástava s úspěšnou resuscitací",
    "I21.4 akutní infarkt myokardu bez elevace ST",
    "I21.0 STEMI přední stěny",
    "I63.9 ischemická CMP",
    "I50.9 dekompensace chronického srdečního selhání",
    "A41.9 sepse",
    "N39.0 infekce močových cest, urosepse",
    "S72.0 pertrochanterická zlomenina femuru po pádu",
    "K80.2 cholelithiáza, cholecystitida",
    "K92.2 krvácení do GIT",
    "J96.0 akutní respirační selhání",
    "E87.1 hyponatrémie, dehydratace",
    "J45.9 těžká exacerbace astmatu",
    "I48.0 nově zachycená fibrilace síní s RVR",
    "N17.9 akutní selhání ledvin",
    "T39.1 intoxikace paracetamolem",
    "S06.5 subdurální hematom po pádu",
    "K35.8 akutní apendicitida",
    "I26.9 suspekce na plicní embolii",
    "E11.6 hyperglykemický stav u DM 2. typu",
    "J44.1 exacerbace CHOPN",
    "R56.0 febrilní křeče / epileptický záchvat",
    "L03.1 erysipel / celulitida dolní končetiny",
    "K59.3 ileózní stav",
]

DOC_TYPES = [
    "propouštěcí zprávu",
    "příjmovou vizitu na ARO",
    "epikrizu (průběh hospitalizace)",
    "odlehčovací souhrn z JIP",
]

# Prompt-randomization pools so generated notes don't share the same skeleton.
OPENING_INSTRUCTIONS = [
    "Dokument začni rovnou textem, bez nadpisu.",
    "První řádek: název typu dokumentu VELKÝMI PÍSMENY (např. PROPOUŠTĚCÍ ZPRÁVA).",
    "První řádek: datum a název oddělení, potom text.",
    "Začni větou o příjmu/převzetí pacienta s datem.",
    "Začni rovnou diagnózou nebo zdůvodněním hospitalizace, údaje pacienta uveď až dál v textu.",
    "Začni stručným souhrnem 'Pacient přijat pro ...'.",
]

STYLE_INSTRUCTIONS = [
    "Piš souvislým odborným textem v odstavcích, bez odrážek.",
    "Rozděl text do sekcí s nadpisy zakončenými dvojtečkou (Anamnéza:, Průběh hospitalizace:, "
    "Medikace:, Doporučení: ...).",
    "Používej telegramový styl a hodně zkratek (obj., klidný, af., vit. fce v normě, kont. dop.).",
    "Piš jako unavený lékař na noční směně — stručně, místy neúplné věty, poznámky v závorkách.",
    "Používej odrážky a čísla dnů hospitalizace (DG 1:, DG 2: ...).",
]

TYPO_INSTRUCTIONS = [
    None,
    None,
    "Text obsahuje 2–5 přirozených překlepů, jaké vznikají při rychlém psaní (přehozená písmena, "
    "vynechané písmeno, špatně napsané slovo — např. 'pacietka', 'Vzdledem', 'snízký').",
    "Piš část textu nebo celý text bez diakritiky, jako starý nemocniční informační systém.",
    "V textu udělej obvyklé chyby sazby: dvojité mezery, chybějící mezera za čárkou nebo tečkou, "
    "občas špatná kapitalizace.",
    "Text obsahuje pár překlepů v dlouhých lékařských termínech a jeden překlep v příjmení "
    "pacienta nebo lékaře (i překleplé jméno stále taguj).",
]

NAME_USAGE = [
    "Jméno pacienta uváděj celé při každém výskytu.",
    "Po prvním uvedení celého jména dál používej jen příjmení ('pacient Novák', 'paní Nováková') — "
    "všechny výskyty taguj.",
    "Jméno pacienta uveď plně jednou, pak používej přízvisko 'pacient', 'pacientka', 'nemocný' "
    "(tyto obecné výrazy NETAGUJ).",
    "Pacienta oslovuj tvarem 'pan/paní + příjmení' skloňovaném podle gramatiky.",
]

DATE_STYLE = [
    "Datum narození a data v dokumentu piš tvarem D.M.YYYY.",
    "Datum narození a data v dokumentu piš tvarem D. M. YYYY (s mezerami).",
    "Data piš tvarem DD.MM.RR (dvoumístný rok).",
]


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _rodne_cislo(year: int, month: int, day: int, female: bool, rng: random.Random) -> str:
    """Plausible rodné číslo; 10-digit form is divisible by 11 (post-1954 rule)."""
    mm = month + (50 if female else 0)
    base = int(f"{year % 100:02d}{mm:02d}{day:02d}") * (1000 if year < 1954 else 10000)
    serial = rng.randrange(1000 if year < 1954 else 10000)
    if year >= 1954:
        serial += (-(base + serial)) % 11
    sep = "/" if year >= 1954 else ""
    return f"{base + serial:09d}" if year < 1954 else f"{base // 10000:06d}{sep}{serial:04d}"


@dataclass
class PatientSeed:
    patient_id: str
    name: str
    female: bool
    age: int
    dob: str
    rc: str
    address: str
    phone: str
    email: str | None
    doctor: str
    hospital: str
    diagnosis: str
    doc_type: str


def sample_patient(patient_id: int, rng: random.Random) -> PatientSeed:
    female = rng.random() < 0.55
    first = rng.choice(FIRST_NAMES_F if female else FIRST_NAMES_M)
    last = rng.choice(LAST_NAMES_F if female else LAST_NAMES_M)
    age = rng.choices(range(18, 99), weights=[max(1, a - 40) for a in range(18, 99)])[0]
    year, month, day = 2026 - age, rng.randint(1, 12), rng.randint(1, 28)
    street, town, psc = rng.choice(ADDRESSES)
    email = (
        f"{_ascii(first).lower()}.{_ascii(last).lower()}{rng.randint(1, 99)}"
        f"@{rng.choice(['seznam.cz', 'email.cz', 'centrum.cz', 'gmail.com'])}"
        if rng.random() < 0.3
        else None
    )
    dr_female = rng.random() < 0.5
    doctor = (
        f"MUDr. {rng.choice(FIRST_NAMES_F if dr_female else FIRST_NAMES_M)} "
        f"{rng.choice(LAST_NAMES_F if dr_female else LAST_NAMES_M)}"
    )
    return PatientSeed(
        patient_id=f"P{patient_id:04d}",
        name=f"{first} {last}",
        female=female,
        age=age,
        dob=f"{day}. {month}. {year}",
        rc=_rodne_cislo(year, month, day, female, rng),
        address=f"{street} {rng.randint(1, 1200)}, {psc} {town}",
        phone=f"+420 {rng.choice(['6', '7'])}{rng.randint(0, 9)}{rng.randint(0, 9)} "
        f"{rng.randint(100, 999)} {rng.randint(100, 999)}",
        email=email,
        doctor=doctor,
        hospital=rng.choice(HOSPITALS),
        diagnosis=rng.choice(DIAGNOSES),
        doc_type=rng.choice(DOC_TYPES),
    )


def build_messages(
    seed: PatientSeed,
    rng: random.Random,
    words: tuple[int, int] = (250, 600),
) -> list[dict]:
    lines = [
        f"- {rng.choice(['Jméno a příjmení', 'Jméno', 'Pacient'])}: {seed.name} "
        f"({'žena' if seed.female else 'muž'})",
        f"- {rng.choice(['Věk', 'věk'])}: {seed.age} let",
        f"- Datum narození: {seed.dob}",
        f"- Rodné číslo: {seed.rc}",
        f"- {rng.choice(['Bydliště', 'Adresa', 'Bydl.'])}: {seed.address}",
        f"- Telefon: {seed.phone}",
        f"- {rng.choice(['Ošetřující lékař', 'Lékař', 'Ošetřoval'])}: {seed.doctor}",
        f"- Nemocnice / oddělení: {seed.hospital}",
        f"- {rng.choice(['Diagnóza', 'Scénář', 'Dg.'])}: {seed.diagnosis}",
    ]
    if seed.email:
        lines.insert(rng.randint(1, len(lines) - 1), f"- E-mail: {seed.email}")
    rng.shuffle(lines)

    # Example opening built from a different patient so it doesn't anchor the output.
    ex_female = rng.random() < 0.5
    ex_name = (
        f"{rng.choice(FIRST_NAMES_F if ex_female else FIRST_NAMES_M)} "
        f"{rng.choice(LAST_NAMES_F if ex_female else LAST_NAMES_M)}"
    )
    ex_street, ex_town, ex_psc = rng.choice(ADDRESSES)
    lo = rng.randint(words[0], (words[0] + words[1]) // 2)
    hi = rng.randint(max(lo + 50, (words[0] + words[1]) // 2), words[1])
    typo = rng.choice(TYPO_INSTRUCTIONS)

    rules = [
        "KAŽDÝ výskyt osobního údaje označ inline XML tagy: <NAME>, <AGE>, <DOB>, <ID>, "
        "<ADDRESS>, <PHONE>, <EMAIL>, <DOCTOR>, <HOSPITAL>.",
        "Jméno taguj vždy, i ve skloňovaném tvaru — „u pana <NAME>Jana Nováka</NAME>\u201c, "
        "„paní <NAME>Marii Novákové</NAME>\u201c — tag zahrnuje celý skloňovaný tvar. "
        "Skloňuj přirozeně, celou větou.",
        "Taguj každé opakování údaje, ne jen první výskyt.",
        "Čísla lůžka, laboratorní hodnoty, data vyšetření a diagnózy NETAGUJ.",
        "Nemusíš použít všechny údaje pacienta — uveď jen ty, které se do dokumentu "
        "přirozeně hodí.",
    ]
    rng.shuffle(rules)

    prompt = f"""Napiš {seed.doc_type} fiktivního pacienta v češtině, jako skutečnou nemocniční \
dokumentaci — odborným lékařským stylem. {rng.choice(STYLE_INSTRUCTIONS)} \
{rng.choice(OPENING_INSTRUCTIONS)} {rng.choice(DATE_STYLE)} {rng.choice(NAME_USAGE)} \
Délka {lo}–{hi} slov. Vypiš pouze text dokumentu, bez komentářů nebo markdownu.
{f'Pozn.: {typo}' if typo else ''}

Údaje pacienta — použij přesně tyto hodnoty:
{chr(10).join(lines)}

Povinná pravidla pro anotaci:
{chr(10).join(f'{i}. {r}' for i, r in enumerate(rules, 1))}

Příklad možného začátku (neopisuj ho doslova):
<NAME>{ex_name}</NAME>, <AGE>{rng.randint(19, 92)}let{'á' if ex_female else 'ý'}</AGE>, \
bydl. <ADDRESS>{ex_street} {rng.randint(1, 500)}, {ex_psc} {ex_town}</ADDRESS>, ..."""
    return [
        {
            "role": "system",
            "content": "Jsi zkušený český lékař. Generuješ syntetickou zdravotní dokumentaci "
            "pro trénink anonymizačního nástroje. Všechny osobní údaje jsou fiktivní.",
        },
        {"role": "user", "content": prompt},
    ]
