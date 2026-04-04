"""
CAESAR Phase 2 — LLM Phishing / Social Engineering Module
===========================================================
Novel Contribution: AI-generated phishing content with controllable attributes.

Two modes:
  1. DEMO mode  — Template-based generation with linguistic perturbation
                  (runs locally, no GPU, no API keys needed)
  2. API mode   — Real HuggingFace / OpenAI generation
                  (requires HUGGINGFACE_TOKEN env variable)

The phishing generator produces emails with:
  • Controllable urgency level (low / medium / high)
  • Attack vector: credential theft, malware delivery, wire transfer fraud
  • Target persona: corporate, banking, IT support
  • Perturbation: character swaps, homoglyphs, typo injection

The Phishing Detector uses bag-of-words TF-IDF + Random Forest
to classify generated emails — demonstrating the arms race.

Output: PhishingResult objects with text + linguistic features for analysis.
"""

from __future__ import annotations

import os
import re
import random
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PhishingEmail:
    subject:       str
    body:          str
    sender:        str
    target_type:   str      # 'corporate' | 'banking' | 'it_support'
    urgency:       str      # 'low' | 'medium' | 'high'
    attack_vector: str      # 'credential' | 'malware' | 'wire_transfer'
    is_phishing:   bool     = True
    perturbed:     bool     = False
    word_count:    int      = 0
    char_count:    int      = 0
    exclamation_n: int      = 0
    url_count:     int      = 0
    urgency_words: int      = 0
    linguistic_score: float = 0.0  # higher = more suspicious

    def __post_init__(self):
        text               = self.subject + ' ' + self.body
        self.word_count    = len(text.split())
        self.char_count    = len(text)
        self.exclamation_n = text.count('!')
        self.url_count     = len(re.findall(r'http[s]?://', text))
        _uw = ['urgent', 'immediately', 'action required', 'verify',
               'suspend', 'expire', 'confirm', 'click', 'password']
        self.urgency_words = sum(1 for w in _uw if w in text.lower())
        self.linguistic_score = self._score()

    def _score(self) -> float:
        s  = min(1.0, self.exclamation_n / 5)   * 0.20
        s += min(1.0, self.url_count)            * 0.25
        s += min(1.0, self.urgency_words / 4)   * 0.35
        s += (1.0 if self.perturbed else 0.0)   * 0.20
        return round(s, 4)

    def full_text(self) -> str:
        return f"From: {self.sender}\nSubject: {self.subject}\n\n{self.body}"


# ═══════════════════════════════════════════════════════════════════════
# Template bank (demo mode)
# ═══════════════════════════════════════════════════════════════════════

_SUBJECTS = {
    'credential': {
        'low':    ["Please verify your account", "Security notice"],
        'medium': ["Your account requires verification", "Login alert"],
        'high':   ["URGENT: Account suspended!", "Action required — immediate"],
    },
    'malware': {
        'low':    ["Shared document awaiting your review"],
        'medium': ["Invoice attached — please review"],
        'high':   ["URGENT: Invoice overdue — open immediately!"],
    },
    'wire_transfer': {
        'low':    ["Payment request pending"],
        'medium': ["Wire transfer needed — CEO request"],
        'high':   ["URGENT: Complete wire transfer NOW"],
    },
}

_BODIES = {
    'credential': {
        'low': (
            "Dear valued customer,\n\n"
            "We have noticed unusual activity on your account. "
            "Please log in to verify your information at the link below.\n\n"
            "http://secure-account-verify.com/login\n\n"
            "Best regards,\nSecurity Team"
        ),
        'medium': (
            "Dear Customer,\n\n"
            "Your account has been flagged for suspicious activity. "
            "To avoid suspension, you must verify your credentials immediately.\n\n"
            "Click here: http://account-verify-secure.net/confirm\n\n"
            "You have 24 hours to complete this action.\n\n"
            "Security Operations Team"
        ),
        'high': (
            "ALERT: Your account will be PERMANENTLY SUSPENDED in 2 hours!\n\n"
            "We detected unauthorised access to your account. "
            "Verify NOW or lose access permanently!\n\n"
            ">>> VERIFY IMMEDIATELY: http://urgent-verify.net/now <<<\n\n"
            "Failure to act will result in account termination.\n\n"
            "— Security Response Team"
        ),
    },
    'malware': {
        'low': (
            "Hi,\n\nI've shared a document with you. "
            "Please download and review at your earliest convenience.\n\n"
            "Download: http://docs-share.net/file.exe\n\nThanks"
        ),
        'medium': (
            "Please find the attached invoice for services rendered. "
            "Payment is due in 30 days.\n\n"
            "Open invoice: http://invoice-portal.com/view.exe\n\n"
            "Finance Department"
        ),
        'high': (
            "FINAL NOTICE: Overdue invoice #INV-9482\n\n"
            "Your account is 60 days overdue. To avoid legal action, "
            "open the invoice and process payment immediately!\n\n"
            "http://invoice-urgent.com/INV9482.exe\n\n"
            "Collections Department"
        ),
    },
    'wire_transfer': {
        'low': (
            "Hi,\n\nCould you please arrange a wire transfer of $15,000 "
            "to a new vendor? Details attached.\n\nThanks,\nManagement"
        ),
        'medium': (
            "This is a confidential request from the CEO. "
            "Please process a wire transfer of $47,000 to the following account "
            "before end of business today. Do not discuss with others.\n\n"
            "Account: IBAN DE89370400440532013000\n\nRegards, CEO"
        ),
        'high': (
            "URGENT — CEO DIRECTIVE\n\n"
            "Process wire of $124,000 IMMEDIATELY. "
            "We are closing a deal and need funds transferred in the next 30 minutes. "
            "Do not reply by email — call me if issues.\n\n"
            "IBAN: GB29NWBK60161331926819\n\n— CEO"
        ),
    },
}

_SENDERS = {
    'corporate':  ['security@corp-notice.com', 'admin@account-alerts.net'],
    'banking':    ['security@banknotify.com', 'alert@mybank-secure.net'],
    'it_support': ['helpdesk@it-support-portal.com', 'noreply@sysadmin-alert.net'],
}

# Homoglyph substitutions for evasion
_HOMOGLYPHS = {
    'a': 'ɑ', 'e': 'е', 'o': '0', 'i': '1', 's': '$', 'l': '1',
}


# ═══════════════════════════════════════════════════════════════════════
# Generator
# ═══════════════════════════════════════════════════════════════════════

class PhishingGenerator:
    """
    LLM-based phishing generator with demo (template) and API modes.

    The generator models the Attacker's social engineering capability.
    Perturbation methods evade keyword-based filters.
    """

    def __init__(self, mode: str = 'demo', seed: int = 42):
        assert mode in ('demo', 'api')
        self.mode = mode
        self.rng  = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        if mode == 'api':
            self._init_api()

    def _init_api(self):
        """Initialise HuggingFace pipeline (requires token)."""
        try:
            from transformers import pipeline
            model = os.environ.get('PHISH_MODEL', 'gpt2')
            self._pipe = pipeline('text-generation', model=model,
                                  max_new_tokens=200)
            print(f"  [Phishing] HuggingFace model loaded: {model}")
        except ImportError:
            print("  [Phishing] transformers not installed — falling back to demo mode")
            self.mode = 'demo'

    def generate(self,
                 attack_vector: str = 'credential',
                 urgency:       str = 'medium',
                 target_type:   str = 'corporate',
                 perturb:       bool = False,
                 n:             int  = 1) -> List[PhishingEmail]:
        """Generate n phishing emails."""
        results = []
        for _ in range(n):
            if self.mode == 'api':
                email = self._api_generate(attack_vector, urgency, target_type)
            else:
                email = self._template_generate(attack_vector, urgency,
                                                target_type, perturb)
            results.append(email)
        return results

    def _template_generate(self, av, urgency, target, perturb) -> PhishingEmail:
        subj_list = _SUBJECTS.get(av, {}).get(urgency, ["Security Notice"])
        subject   = self.rng.choice(subj_list)
        body      = _BODIES.get(av, {}).get(urgency, "")
        sender    = self.rng.choice(_SENDERS.get(target, ['admin@fake.com']))

        if perturb:
            body   = self._perturb(body)
            sender = self._perturb_domain(sender)

        return PhishingEmail(
            subject=subject, body=body, sender=sender,
            target_type=target, urgency=urgency,
            attack_vector=av, perturbed=perturb,
        )

    def _api_generate(self, av, urgency, target) -> PhishingEmail:
        """Real LLM generation via HuggingFace."""
        prompt = (f"Write a {urgency}-urgency {av.replace('_',' ')} phishing email "
                  f"targeting a {target.replace('_',' ')} employee:\n\n")
        out  = self._pipe(prompt)[0]['generated_text']
        body = out[len(prompt):]
        return PhishingEmail(
            subject=f"[AI-generated] {av}",
            body=body,
            sender='ai@generated.com',
            target_type=target, urgency=urgency,
            attack_vector=av, perturbed=False,
        )

    # ── Perturbation methods ───────────────────────────────────────────
    def _perturb(self, text: str, rate: float = 0.03) -> str:
        """Apply homoglyph substitution to evade keyword filters."""
        chars = list(text)
        for i, c in enumerate(chars):
            if c.lower() in _HOMOGLYPHS and self.rng.random() < rate:
                chars[i] = _HOMOGLYPHS[c.lower()]
        return ''.join(chars)

    def _perturb_domain(self, email: str) -> str:
        """Slightly alter domain to evade domain-based filters."""
        local, _, domain = email.partition('@')
        parts  = domain.split('.')
        if parts:
            parts[0] += self.rng.choice(['-secure', '-alert', '-notice', '1'])
        return f"{local}@{'.'.join(parts)}"

    def generate_dataset(self, n_phish: int = 200,
                         n_legit: int = 200) -> List[Dict]:
        """Generate balanced phishing + legitimate email dataset."""
        vectors   = ['credential', 'malware', 'wire_transfer']
        urgencies = ['low', 'medium', 'high']
        targets   = ['corporate', 'banking', 'it_support']

        records = []

        # Phishing
        for _ in range(n_phish):
            e = self._template_generate(
                av      = self.rng.choice(vectors),
                urgency = self.rng.choice(urgencies),
                target  = self.rng.choice(targets),
                perturb = self.rng.random() > 0.5,
            )
            records.append({
                'text':            e.full_text(),
                'label':           1,
                'urgency':         e.urgency,
                'attack_vector':   e.attack_vector,
                'perturbed':       e.perturbed,
                'ling_score':      e.linguistic_score,
                'word_count':      e.word_count,
                'exclamation_n':   e.exclamation_n,
                'url_count':       e.url_count,
                'urgency_words':   e.urgency_words,
            })

        # Legitimate
        legit_templates = [
            ("Team meeting tomorrow at 10am. Please confirm attendance.",
             "Hi team, just a reminder about our weekly sync. Agenda attached."),
            ("Q3 report available for review",
             "Dear colleagues, the Q3 financial report is now available on SharePoint."),
            ("Welcome to our newsletter",
             "Thank you for subscribing. Here are this month's industry updates."),
            ("Project update — Phase 2 complete",
             "Hi all, pleased to report that Phase 2 is complete. Moving to Phase 3."),
        ]
        for _ in range(n_legit):
            subj, body = self.rng.choice(legit_templates)
            records.append({
                'text':          f"Subject: {subj}\n\n{body}",
                'label':         0,
                'urgency':       'low',
                'attack_vector': 'none',
                'perturbed':     False,
                'ling_score':    round(self._np_rng.uniform(0, 0.15), 4),
                'word_count':    len(body.split()),
                'exclamation_n': 0,
                'url_count':     0,
                'urgency_words': 0,
            })

        self.rng.shuffle(records)
        return records


# ═══════════════════════════════════════════════════════════════════════
# Phishing Detector (the Defender side)
# ═══════════════════════════════════════════════════════════════════════

class PhishingDetector:
    """
    TF-IDF + Random Forest phishing classifier (Defender).

    Trained on generated emails; tested on perturbed ones to measure
    robustness against evasion — a clean adversarial ML experiment.
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline

        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=1000, ngram_range=(1, 2),
                sublinear_tf=True, strip_accents='unicode',
            )),
            ('clf', RandomForestClassifier(
                n_estimators=100, max_depth=15,
                class_weight='balanced', random_state=42,
            )),
        ])
        self.fitted = False

    def fit(self, texts: List[str], labels: List[int]) -> 'PhishingDetector':
        print("  [Phishing Detector] Training on generated email dataset...")
        self.pipeline.fit(texts, labels)
        self.fitted = True
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)[:, 1]

    def evaluate(self, texts: List[str], labels: List[int]) -> Dict:
        from sklearn.metrics import (accuracy_score, f1_score,
                                     roc_auc_score, confusion_matrix)
        y_pred = self.predict(texts)
        y_prob = self.predict_proba(texts)
        cm     = confusion_matrix(labels, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,cm[0,0])
        return {
            'accuracy':       float(accuracy_score(labels, y_pred)),
            'f1':             float(f1_score(labels, y_pred, zero_division=0)),
            'roc_auc':        float(roc_auc_score(labels, y_prob)),
            'detection_rate': float(tp / (tp + fn + 1e-9)),
            'false_pos_rate': float(fp / (fp + tn + 1e-9)),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        }

    def top_phishing_terms(self, top_k: int = 20) -> List[Tuple[str, float]]:
        """Return the most discriminative phishing terms."""
        if not self.fitted:
            return []
        tfidf = self.pipeline.named_steps['tfidf']
        clf   = self.pipeline.named_steps['clf']
        feat_names = tfidf.get_feature_names_out()
        imps       = clf.feature_importances_
        idx        = np.argsort(imps)[::-1][:top_k]
        return [(feat_names[i], float(imps[i])) for i in idx]
