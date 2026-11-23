"""Test cases shared by the reference runner (PyTorch, original code), the MLX
verifier and the benchmarks. Torch-free and MLX-free, so both environments import it.

The first part is copied from ../compare.py (the 118 accuracy questions); the
second adds paths compare.py does not touch: several questions over one state
(the d1-3B trunk/branch tree, d1-2B set attention), JSON and long states, and
d1-2B span reads (text, number, date), estimates and multiple choice.
"""
import random

# ---------------------------------------------------------------- 質問定義
TEAM = {
    "type": "choice",
    "instructions": "Which team should handle this support ticket?",
    "criteria": {
        "billing": "Payments, charges, invoices, refunds, subscriptions.",
        "engineering": "Bugs, crashes, errors, outages, technical failures.",
        "sales": "Pricing questions, upgrades, quotes, buying more seats.",
        "account": "Login, password reset, profile, deleting or changing the account.",
    },
}
SENTIMENT = {
    "type": "choice",
    "instructions": "What is the sentiment of this product review?",
    "criteria": {"positive": "The reviewer is satisfied.", "negative": "The reviewer is dissatisfied.",
                 "neutral": "Mixed or no clear opinion."},
}
URGENCY = {
    "type": "score",
    "instructions": "How urgent is this message?",
    "criteria": ["Low: can wait days.", "Medium: should be handled today.", "High: needs immediate action."],
}


def noul(q):
    return {"type": "noul", "instructions": q}


# (state, question, 正解)  choice→ラベル, noul→True/False, score→レベル番号
TEXT_CASES = [
    # --- 担当チーム振り分け
    ("I was charged twice for my monthly plan. Please refund the duplicate payment.", TEAM, "billing"),
    ("Where can I download last month's invoice as a PDF?", TEAM, "billing"),
    ("My credit card expired, how do I update the payment method for my subscription?", TEAM, "billing"),
    ("I cancelled my subscription but you still billed me this month.", TEAM, "billing"),
    ("The app crashes every time I open the settings screen on Android 15.", TEAM, "engineering"),
    ("Exporting a report returns a 500 Internal Server Error.", TEAM, "engineering"),
    ("Your API has been timing out for all requests since 3am.", TEAM, "engineering"),
    ("Uploaded images show up as blank grey boxes in the dashboard.", TEAM, "engineering"),
    ("We are a team of 200 people. Can you give us a quote for the enterprise plan?", TEAM, "sales"),
    ("What is the price difference between the Pro and Business tiers?", TEAM, "sales"),
    ("We want to buy 15 more seats for our organization.", TEAM, "sales"),
    ("Do you offer discounts for non-profit organizations?", TEAM, "sales"),
    ("I forgot my password and the reset email never arrives.", TEAM, "account"),
    ("How do I change the email address on my profile?", TEAM, "account"),
    ("Please delete my account and all my personal data.", TEAM, "account"),
    ("I can't log in, it says my account is locked after too many attempts.", TEAM, "account"),
    # --- 感情分類
    ("Absolutely love it. Battery lasts two days and the screen is gorgeous.", SENTIMENT, "positive"),
    ("Best purchase I've made this year, works perfectly out of the box.", SENTIMENT, "positive"),
    ("Fast shipping, great quality, would buy again.", SENTIMENT, "positive"),
    ("Broke after one week. Customer service ignored my emails. Avoid.", SENTIMENT, "negative"),
    ("Terrible sound quality and it hurts my ears. Returning it.", SENTIMENT, "negative"),
    ("Total waste of money, the zipper fell off on day one.", SENTIMENT, "negative"),
    ("It's a phone case. It fits the phone. Nothing more to say.", SENTIMENT, "neutral"),
    ("The color is nice but the material feels cheap; overall it's okay.", SENTIMENT, "neutral"),
    ("Arrived on Tuesday. Haven't used it much yet.", SENTIMENT, "neutral"),
    # --- 緊急度 (0=低,1=中,2=高)
    ("Production database is down and customers cannot check out. Revenue is being lost every minute.", URGENCY, 2),
    ("Security alert: someone is logging into our admin account from an unknown country right now.", URGENCY, 2),
    ("The payment page is broken for all users since the deploy 10 minutes ago.", URGENCY, 2),
    ("Our client demo is in two hours and the report generator is failing. Can someone look today?", URGENCY, 1),
    ("A few users report slow page loads this afternoon; please investigate before end of day.", URGENCY, 1),
    ("The weekly invoice job failed; it needs to run before tomorrow morning.", URGENCY, 1),
    ("Small typo on the About page: 'recieve' should be 'receive'. No rush.", URGENCY, 0),
    ("Feature idea for someday: a dark mode for the settings page.", URGENCY, 0),
    ("Whenever you have time next month, could you update the team photo on the website?", URGENCY, 0),
    # --- yes/no
    ("Order #4821: 3 items shipped on May 2, delivered May 5. Status: delivered.", noul("Has the order been delivered?"), True),
    ("Order #4822: payment received, awaiting shipment.", noul("Has the order been delivered?"), False),
    ("Hi, could you please refund my last payment? I didn't mean to renew.", noul("Is the customer asking for a refund?"), True),
    ("Hi, how do I export my data to CSV?", noul("Is the customer asking for a refund?"), False),
    ('{"user": "kim", "age": 17, "country": "JP"}', noul("Is the user an adult (18 or older)?"), False),
    ('{"user": "lee", "age": 34, "country": "KR"}', noul("Is the user an adult (18 or older)?"), True),
    ("Meeting moved from Thursday 3pm to Friday 10am.", noul("Is the meeting on Friday?"), True),
    ("Meeting confirmed for Thursday at 3pm in room B.", noul("Is the meeting on Friday?"), False),
    ("Click here to claim your $1000 prize! Enter your bank password to verify.", noul("Is this message a phishing or scam attempt?"), True),
    ("Reminder: the team lunch is at noon tomorrow in the cafeteria.", noul("Is this message a phishing or scam attempt?"), False),
    ("The invoice total is $1,250 and it is due on June 30.", noul("Is the invoice total more than $1,000?"), True),
    ("The invoice total is $480 and it is due on June 30.", noul("Is the invoice total more than $1,000?"), False),
]

# ---------------------------------------------------------------- 画像問題 (PIL で描画)
COLORS = {"red": (220, 40, 40), "blue": (40, 80, 220), "green": (40, 170, 70), "yellow": (240, 200, 30)}
COUNT_Q = {"type": "choice", "instructions": "How many shapes are in the picture?",
           "criteria": {"one": "Exactly one shape.", "two": "Exactly two shapes.",
                        "three": "Exactly three shapes.", "four": "Exactly four shapes."}}
COLOR_Q = {"type": "choice", "instructions": "What color are the shapes?",
           "criteria": {k: f"The shapes are {k}." for k in COLORS}}
SHAPE_Q = {"type": "choice", "instructions": "What kind of shape is shown?",
           "criteria": {"circle": "Circles.", "square": "Squares.", "triangle": "Triangles."}}
NUM_WORDS = ["one", "two", "three", "four"]


def draw(n, color, shape, seed):
    from PIL import Image, ImageDraw
    rnd = random.Random(seed)
    img = Image.new("RGB", (384, 384), "white")
    d = ImageDraw.Draw(img)
    slots = [(40, 40), (220, 40), (40, 220), (220, 220)]
    rnd.shuffle(slots)
    for x, y in slots[:n]:
        x += rnd.randint(0, 20); y += rnd.randint(0, 20)
        box = (x, y, x + 110, y + 110)
        if shape == "circle":
            d.ellipse(box, fill=COLORS[color])
        elif shape == "square":
            d.rectangle(box, fill=COLORS[color])
        else:
            d.polygon([(x + 55, y), (x, y + 110), (x + 110, y + 110)], fill=COLORS[color])
    return img


def image_cases():
    rnd = random.Random(0)
    cases = []
    for i in range(24):
        n, color, shape = rnd.randint(1, 4), rnd.choice(list(COLORS)), rnd.choice(["circle", "square", "triangle"])
        img = draw(n, color, shape, i)
        cases.append((img, {"count": COUNT_Q, "color": COLOR_Q, "shape": SHAPE_Q},
                      {"count": NUM_WORDS[n - 1], "color": color, "shape": shape}))
    return cases



def pred(ans):
    """The predicted label of one answer (compare.py's rule)."""
    t = ans["type"]
    if t == "choice":
        return ans["choice"]
    if t == "noul":
        return ans["noul"] >= 0.5
    if t == "score":  # 3B returns the expected level, 2B the median: compare the likeliest level
        p = ans["probabilities"]
        return int(max(p, key=p.get))
    raise ValueError(t)


def accuracy_cases():
    """(case id, state, {name: question}, image or None, {name: gold}) for compare.py's 118 questions."""
    out = []
    for i, (state, q, gold) in enumerate(TEXT_CASES):
        kind = q["type"] if q is not TEAM and q is not SENTIMENT else ("team" if q is TEAM else "sentiment")
        out.append((f"text{i:02d}:{kind}", state, {"q": q}, None, {"q": gold}))
    for i, (img, qs, gold) in enumerate(image_cases()):
        out.append((f"image{i:02d}", None, qs, img, gold))
    return out


# ---------------------------------------------------------------- extra paths
TICKET = ("I was charged twice for my subscription this month and the app keeps crashing "
          "when I open settings. Please fix this ASAP, I need it for work tomorrow.")
THREE = {"team": TEAM, "urgency": URGENCY, "refund": noul("Is the customer asking for a refund?")}
FIVE_LEVELS = {"type": "score", "instructions": "How satisfied is the customer?",
               "criteria": ["Furious.", "Unhappy.", "Neutral.", "Happy.", "Delighted."]}
NOUL_CRITERIA = {"type": "noul", "instructions": "Does the message mention money?",
                 "criteria": {"true": "Charges, prices or refunds are mentioned.", "false": "No money is mentioned."}}
ORDER = {"order": {"id": "A-1042", "customer": {"name": "Mika Tanaka", "tier": "gold"},
                   "items": [{"sku": "KB-01", "qty": 2, "price": 49.5}, {"sku": "MS-07", "qty": 1, "price": 25}],
                   "status": "shipped", "shipped_on": "2026-03-14", "notes": "Leave at the front desk."}}


def extra_cases(model: str):
    """Cases beyond compare.py, as (case id, state, {name: question}, image or None)."""
    long_state = " ".join(s for s, _, _ in TEXT_CASES) * 6
    img = image_cases()[1][0]
    out = [
        ("tree3:text", TICKET, THREE, None),
        ("tree5:text", TICKET, {**THREE, "money": NOUL_CRITERIA, "mood": FIVE_LEVELS}, None),
        ("tree3:image", None, image_cases()[1][1], img),
        ("tree2:image+text", "A photo sent by a customer with their order.",
         {"count": COUNT_Q, "refund": noul("Is the customer asking for a refund?")}, img),
        ("json:noul", ORDER, {"q": noul("Has the order been shipped?")}, None),
        ("json:tree", ORDER, {"tier": {"type": "choice", "instructions": "Which loyalty tier is the customer?",
                                       "criteria": {"bronze": None, "silver": None, "gold": None}},
                              "big": noul("Is the order total above 100 dollars?")}, None),
        ("long:choice", long_state, {"team": TEAM}, None),
        ("long:tree", long_state, THREE, None),
        ("score5", "Thanks, the replacement arrived early and works great!", {"q": FIVE_LEVELS}, None),
        ("noul:criteria", TICKET, {"q": NOUL_CRITERIA}, None),
        ("choice:letters", "The answer is the second option.",
         {"q": {"type": "choice", "instructions": "Which option does the text point to?",
                "criteria": {"A": "First option.", "B": "Second option.", "C": "Third option."}}}, None),
    ]
    if model.endswith("2B"):
        out += [
            ("read:text", ORDER, {"name": {"type": "text", "instructions": "What is the customer's name?"},
                                  "note": {"type": "text", "instructions": "What delivery note was left?"}}, None),
            ("read:number", ORDER, {"qty": {"type": "number", "instructions": "How many KB-01 keyboards were ordered?"},
                                    "price": {"type": "number", "instructions": "What is the unit price of MS-07?",
                                              "unit": "USD"}}, None),
            ("read:date", ORDER, {"shipped": {"type": "date", "instructions": "When was the order shipped?"}}, None),
            ("read:text-plain", TICKET, {"q": {"type": "text", "instructions": "What part of the app crashes?"}}, None),
            ("estimate", TICKET, {"q": {"type": "number", "instructions": "How angry is the customer from 0 to 10?",
                                        "range": [0, 10], "step": 1}}, None),
            ("choice:multiple", TICKET, {"q": {"type": "choice", "multiple": True,
                                               "instructions": "Which problems does the customer report?",
                                               "criteria": {"billing": "A wrong charge.", "crash": "The app crashes.",
                                                            "login": "Cannot log in.", "slow": "The app is slow."}}}, None),
            ("mixed:read+typed", ORDER, {"name": {"type": "text", "instructions": "What is the customer's name?"},
                                         "shipped": noul("Has the order been shipped?"), "tier": THREE["team"]}, None),
        ]
    return out


def all_cases(model: str):
    """Every case as (case id, state, questions, image or None)."""
    return [c[:4] for c in accuracy_cases()] + extra_cases(model)
