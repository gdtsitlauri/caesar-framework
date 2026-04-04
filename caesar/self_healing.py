"""
CAESAR Phase 3 — Self-Healing Security System
==============================================
Novel Contribution #6: Autonomous Remediation Loop (ARL)

The Self-Healing System extends CAESAR with a closed-loop autonomy layer:

  Detect → Classify → Select Countermeasure → Deploy → Verify → Adapt

Key properties:
  • Zero-touch remediation for known attack patterns
  • Escalation protocol for novel/high-severity threats
  • Countermeasure effectiveness tracking + auto-rotation
  • Healing confirmation via post-action health diff

State machine:
  NORMAL → ALERT → RESPONDING → HEALING → VERIFIED → NORMAL

This is the "self-healing AI security systems" mentioned in the thesis proposal
and represents the forward-looking Chapter 5 contribution.
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np


# ── System states ──────────────────────────────────────────────────────
class SystemState(IntEnum):
    NORMAL     = 0
    ALERT      = 1
    RESPONDING = 2
    HEALING    = 3
    VERIFIED   = 4


STATE_NAMES = ['NORMAL', 'ALERT', 'RESPONDING', 'HEALING', 'VERIFIED']

ATTACK_NAMES  = ['NORMAL','DOS','DDOS','PORT_SCAN','BRUTE_FORCE',
                 'DATA_EXFIL','MITM','RANSOMWARE']
DEFENSE_NAMES = ['NO_ACTION','BLOCK_IP','RATE_LIMIT','ISOLATE_NODE',
                 'DEPLOY_HONEYPOT','ALERT_ADMIN','PATCH_VULN','RESET_CONN']

# Severity mapping
SEVERITY = {0: 0.0, 1: 0.6, 2: 0.9, 3: 0.3, 4: 0.5,
            5: 0.7, 6: 0.8, 7: 1.0}


# ── Healing event record ───────────────────────────────────────────────
@dataclass
class HealingEvent:
    step:            int
    attack_type:     int
    severity:        float
    state_before:    int
    action_taken:    int
    health_before:   float
    health_after:    float
    success:         bool
    time_to_heal_ms: float
    escalated:       bool = False

    @property
    def health_delta(self) -> float:
        return self.health_after - self.health_before


# ═══════════════════════════════════════════════════════════════════════
class SelfHealingSystem:
    """
    Autonomous Remediation Loop (ARL) — wraps the CAESAR framework
    with a state machine for zero-touch incident response.
    """

    # Thresholds
    ALERT_THRESHOLD    = 0.30   # anomaly score that triggers ALERT
    RESPOND_THRESHOLD  = 0.50   # severity that triggers auto-response
    VERIFY_ROUNDS      = 3      # rounds needed to confirm healing
    ESCALATE_SEVERITY  = 0.85   # severity that escalates to human

    def __init__(self, caesar, env):
        self.caesar  = caesar
        self.env     = env
        self.state   = SystemState.NORMAL

        # Countermeasure rotation: track effectiveness per (attack, action)
        self._cm_success:  Dict[Tuple[int,int], list] = {}
        self._cm_failures: Dict[int, int] = {}   # action → consecutive failures

        # Healing history
        self.events:         List[HealingEvent] = []
        self.state_history:  List[int]          = []
        self.health_history: List[float]        = []

        # Verify countdown
        self._verify_count:  int           = 0
        self._active_action: Optional[int] = None
        self._active_attack: Optional[int] = None
        self._health_snap:   float         = 1.0
        self._step:          int           = 0

    # ── State machine core ───────────────────────────────────────────
    def tick(self, attack_type: int, attack_success: float) -> Dict:
        """
        Process one simulation tick through the healing state machine.
        Returns event dict describing what happened this tick.
        """
        self._step += 1
        severity  = SEVERITY.get(attack_type, 0.5) * attack_success
        anomaly   = float(self.env.state.get('anomaly', attack_success))
        health    = self.env.health()
        self.health_history.append(health)

        event = {
            'step':         self._step,
            'attack_type':  attack_type,
            'severity':     severity,
            'health':       health,
            'system_state': self.state.name,
            'action':       'none',
            'escalated':    False,
            'healed':       False,
        }

        # ── State transitions ────────────────────────────────────────
        if self.state == SystemState.NORMAL:
            if anomaly > self.ALERT_THRESHOLD or severity > 0.1:
                self.state = SystemState.ALERT
                event['action'] = 'entered_alert'

        elif self.state == SystemState.ALERT:
            if severity >= self.RESPOND_THRESHOLD:
                self.state   = SystemState.RESPONDING
                self._health_snap = health
                event['action'] = 'auto_responding'

                if severity >= self.ESCALATE_SEVERITY:
                    event['escalated'] = True
                    event['action']    = 'escalated_to_human'
            elif anomaly < self.ALERT_THRESHOLD * 0.5:
                self.state  = SystemState.NORMAL
                event['action'] = 'threat_cleared'

        elif self.state == SystemState.RESPONDING:
            # Choose best countermeasure via TAG + ADPN
            action = self._select_countermeasure(attack_type, health)
            self._active_action = action
            self._active_attack = attack_type

            def_reward, _ = self.env.defend(action)
            new_health    = self.env.health()

            # Update effectiveness memory
            key = (attack_type, action)
            self._cm_success.setdefault(key, []).append(def_reward)

            if def_reward > 0.2:
                self.state         = SystemState.HEALING
                self._verify_count = 0
                event['action']    = f'deployed_{DEFENSE_NAMES[action]}'
                event['healed']    = (new_health > health)
            else:
                # Countermeasure ineffective → try rotation
                self._cm_failures[action] = self._cm_failures.get(action, 0) + 1
                alt = self._rotate_countermeasure(attack_type, action)
                self._active_action = alt
                def_reward2, _ = self.env.defend(alt)
                new_health     = self.env.health()
                event['action'] = f'rotated_to_{DEFENSE_NAMES[alt]}'

                if def_reward2 > 0.1:
                    self.state         = SystemState.HEALING
                    self._verify_count = 0

        elif self.state == SystemState.HEALING:
            health_now = self.env.health()
            if health_now > self._health_snap - 0.05:
                self._verify_count += 1
                event['action'] = f'verifying_{self._verify_count}/{self.VERIFY_ROUNDS}'
                if self._verify_count >= self.VERIFY_ROUNDS:
                    self.state    = SystemState.VERIFIED
                    event['healed'] = True
                    self._log_healing(health, health_now)
            else:
                # Healing failed — re-enter RESPONDING
                self.state  = SystemState.RESPONDING
                event['action'] = 'healing_failed_retry'

        elif self.state == SystemState.VERIFIED:
            # Decay back to NORMAL after one clean tick
            self.state   = SystemState.NORMAL
            event['action'] = 'verified_and_recovered'

        self.state_history.append(int(self.state))
        return event

    # ── Countermeasure intelligence ──────────────────────────────────
    def _select_countermeasure(self, attack_type: int, health: float) -> int:
        """Select countermeasure: TAG history first, ADPN fallback."""
        # Use TAG's best defense if enough history
        best = self.caesar.tag.best_defense(attack_type)
        key  = (attack_type, best)
        hist = self._cm_success.get(key, [])
        if hist and np.mean(hist) > 0.3:
            return best

        # Fallback: ADPN Q-values
        state_vec = self.env._vec()
        return self.caesar.adpn.select_action(state_vec)

    def _rotate_countermeasure(self, attack_type: int, failed: int) -> int:
        """
        Countermeasure rotation: pick next best that hasn't recently failed.
        """
        # Get all defenses sorted by historical effectiveness
        candidates = []
        for d in range(8):
            if d == failed: continue
            failures = self._cm_failures.get(d, 0)
            key      = (attack_type, d)
            avg_rew  = np.mean(self._cm_success[key]) if key in self._cm_success else 0.1
            score    = avg_rew - 0.15 * failures
            candidates.append((score, d))
        candidates.sort(reverse=True)
        return candidates[0][1] if candidates else 0

    def _log_healing(self, health_before: float, health_after: float):
        if self._active_action is None: return
        ev = HealingEvent(
            step=self._step,
            attack_type=self._active_attack or 0,
            severity=SEVERITY.get(self._active_attack or 0, 0.5),
            state_before=SystemState.RESPONDING,
            action_taken=self._active_action,
            health_before=health_before,
            health_after=health_after,
            success=(health_after >= health_before),
            time_to_heal_ms=self._verify_count * 10.0,
            escalated=(self._cm_failures.get(self._active_action, 0) > 0),
        )
        self.events.append(ev)

    # ── Run simulation ────────────────────────────────────────────────
    def run_simulation(self, n_ticks: int = 200, verbose: bool = True) -> List[Dict]:
        """Run the self-healing simulation for n_ticks."""
        print(f"\n  [SelfHealing] Running ARL simulation ({n_ticks} ticks)...")
        self.env.reset()
        log = []

        for tick in range(n_ticks):
            # Inject attack from TAG-aware CAESAR
            d_emb    = self.env.defense_embed()
            atk, ins = self.caesar.tagan.generate_attack(d_emb)
            a_succ   = self.env.inject(atk, ins)

            event = self.tick(atk, a_succ)
            log.append(event)

            if verbose and tick % 40 == 0:
                print(f"    tick {tick:4d} | State: {event['system_state']:<12s}"
                      f"| Health: {event['health']:.3f}"
                      f"| Severity: {event['severity']:.3f}"
                      f"| Action: {event['action']}")

        n_healed  = sum(1 for e in self.events)
        n_escal   = sum(1 for e in log if e['escalated'])
        avg_h     = np.mean(self.health_history)
        print(f"\n  [SelfHealing] Complete.")
        print(f"    Healing events : {n_healed}")
        print(f"    Escalations    : {n_escal}")
        print(f"    Avg Health     : {avg_h:.3f}")
        return log

    # ── Metrics ──────────────────────────────────────────────────────
    def summary(self) -> Dict:
        state_arr = np.array(self.state_history)
        return {
            'total_ticks':         len(self.state_history),
            'healing_events':      len(self.events),
            'mean_health':         float(np.mean(self.health_history)) if self.health_history else 0.,
            'pct_normal':          float((state_arr == 0).mean()) if len(state_arr) else 0.,
            'pct_responding':      float((state_arr == 2).mean()) if len(state_arr) else 0.,
            'pct_healing':         float((state_arr == 3).mean()) if len(state_arr) else 0.,
            'successful_heals':    sum(1 for e in self.events if e.success),
            'avg_time_to_heal_ms': float(np.mean([e.time_to_heal_ms for e in self.events])) if self.events else 0.,
        }
