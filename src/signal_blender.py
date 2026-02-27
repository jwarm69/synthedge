"""
Signal blender for combining SynthData and local ensemble predictions.

Key insight: when 200+ independent Bittensor models AND our local ensemble
both agree on direction, the combined signal is much stronger than either alone.
When they disagree, we sit out.
"""

from enum import Enum
from typing import Optional


class Agreement(Enum):
    STRONG_AGREE = "STRONG_AGREE"
    AGREE = "AGREE"
    NEUTRAL = "NEUTRAL"
    DISAGREE = "DISAGREE"
    STRONG_DISAGREE = "STRONG_DISAGREE"


class EdgeQuality(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# Default weights for 1h horizon (both sources available)
WEIGHT_SYNTHDATA_1H = 0.45
WEIGHT_ENSEMBLE_1H = 0.55

# For 15min (no local ensemble, use 1h ensemble direction as context)
WEIGHT_SYNTHDATA_15MIN = 0.80
WEIGHT_ENSEMBLE_CONTEXT_15MIN = 0.20

# Agreement boost parameters
MAX_AGREEMENT_BOOST = 0.15
AGREEMENT_STRENGTH_THRESHOLD = 0.55  # both must exceed this for boost


def blend_predictions(
    synthdata_prob_up: float,
    ensemble_prob_up: float,
    method: str = "agreement_boost",
    horizon: str = "1h",
) -> dict:
    """
    Blend SynthData and local ensemble probabilities.

    Args:
        synthdata_prob_up: P(UP) from SynthData (0-1)
        ensemble_prob_up: P(UP) from local ensemble (0-1)
        method: "agreement_boost" (default) or "weighted_average"
        horizon: "1h" or "15min"

    Returns:
        {
            "blended_prob_up": float,
            "blended_direction": "UP"|"DOWN",
            "blended_confidence": float,  # 0-1 scale
            "agreement": Agreement enum,
            "agreement_score": float,  # -1 to 1
            "quality": EdgeQuality enum,
            "synthdata_prob_up": float,
            "ensemble_prob_up": float,
            "boost_applied": float,
            "method": str,
        }
    """
    if horizon in ("15min", "15m"):
        w_synth = WEIGHT_SYNTHDATA_15MIN
        w_ens = WEIGHT_ENSEMBLE_CONTEXT_15MIN
    else:
        w_synth = WEIGHT_SYNTHDATA_1H
        w_ens = WEIGHT_ENSEMBLE_1H

    # Base weighted average
    base_prob = w_synth * synthdata_prob_up + w_ens * ensemble_prob_up

    # Classify agreement
    agreement = classify_agreement(synthdata_prob_up, ensemble_prob_up)
    agreement_score = _agreement_score(synthdata_prob_up, ensemble_prob_up)

    boost = 0.0

    if method == "agreement_boost":
        synth_dir = "UP" if synthdata_prob_up > 0.5 else "DOWN"
        ens_dir = "UP" if ensemble_prob_up > 0.5 else "DOWN"

        if synth_dir == ens_dir:
            # Both agree - apply confidence boost
            synth_strength = abs(synthdata_prob_up - 0.5) * 2
            ens_strength = abs(ensemble_prob_up - 0.5) * 2

            if (synthdata_prob_up > AGREEMENT_STRENGTH_THRESHOLD and
                    ensemble_prob_up > AGREEMENT_STRENGTH_THRESHOLD):
                # Strong agreement on UP
                boost = min(synth_strength * ens_strength, 1.0) * MAX_AGREEMENT_BOOST
            elif (synthdata_prob_up < (1 - AGREEMENT_STRENGTH_THRESHOLD) and
                    ensemble_prob_up < (1 - AGREEMENT_STRENGTH_THRESHOLD)):
                # Strong agreement on DOWN
                boost = -min(synth_strength * ens_strength, 1.0) * MAX_AGREEMENT_BOOST
            else:
                # Weak agreement - smaller boost in agreed direction
                if synth_dir == "UP":
                    boost = min(synth_strength * ens_strength, 1.0) * MAX_AGREEMENT_BOOST * 0.5
                else:
                    boost = -min(synth_strength * ens_strength, 1.0) * MAX_AGREEMENT_BOOST * 0.5
        else:
            # Disagree - pull toward 0.5 (reduce confidence)
            pull_strength = min(
                abs(synthdata_prob_up - 0.5),
                abs(ensemble_prob_up - 0.5)
            )
            boost = (0.5 - base_prob) * pull_strength * 2

    blended_prob = max(0.01, min(0.99, base_prob + boost))

    direction = "UP" if blended_prob > 0.5 else "DOWN"
    confidence = abs(blended_prob - 0.5) * 2

    # Compute quality
    vol_ratio = 1.0  # default, can be overridden by caller
    quality = compute_edge_quality(agreement_score, confidence, vol_ratio)

    return {
        "blended_prob_up": blended_prob,
        "blended_direction": direction,
        "blended_confidence": confidence,
        "agreement": agreement,
        "agreement_score": agreement_score,
        "quality": quality,
        "synthdata_prob_up": synthdata_prob_up,
        "ensemble_prob_up": ensemble_prob_up,
        "boost_applied": boost,
        "method": method,
        "horizon": horizon,
    }


def blend_synthdata_only(
    synthdata_prob_up: float,
    ensemble_direction_1h: Optional[str] = None,
    horizon: str = "15min",
) -> dict:
    """
    Blend when only SynthData is available (15min markets).

    Uses 1h ensemble direction as weak context signal.
    """
    # If ensemble direction available, create a weak probability from it
    if ensemble_direction_1h == "UP":
        ens_context_prob = 0.55  # weak UP bias
    elif ensemble_direction_1h == "DOWN":
        ens_context_prob = 0.45  # weak DOWN bias
    else:
        ens_context_prob = 0.50  # no context

    return blend_predictions(
        synthdata_prob_up=synthdata_prob_up,
        ensemble_prob_up=ens_context_prob,
        method="agreement_boost",
        horizon=horizon,
    )


def classify_agreement(p_synth: float, p_ens: float) -> Agreement:
    """
    Classify the level of agreement between two probability sources.

    Both sources express probability as P(UP), so:
    - Both > 0.6 or both < 0.4 = strong agreement
    - Both same side of 0.5 = agreement
    - One > 0.5, other < 0.5 = disagreement
    """
    synth_up = p_synth > 0.5
    ens_up = p_ens > 0.5

    synth_strength = abs(p_synth - 0.5)
    ens_strength = abs(p_ens - 0.5)

    if synth_up == ens_up:
        # Same direction
        if synth_strength > 0.10 and ens_strength > 0.10:
            return Agreement.STRONG_AGREE
        elif synth_strength > 0.03 and ens_strength > 0.03:
            return Agreement.AGREE
        else:
            return Agreement.NEUTRAL
    else:
        # Opposite directions
        if synth_strength > 0.10 and ens_strength > 0.10:
            return Agreement.STRONG_DISAGREE
        elif synth_strength > 0.03 and ens_strength > 0.03:
            return Agreement.DISAGREE
        else:
            return Agreement.NEUTRAL


def _agreement_score(p_synth: float, p_ens: float) -> float:
    """
    Numerical agreement score from -1 (strong disagree) to +1 (strong agree).

    Positive = both sources agree and are confident.
    Negative = sources disagree.
    Zero = neutral / mixed signals.
    """
    # Direction alignment: +1 if same, -1 if different
    synth_dir = 1 if p_synth > 0.5 else -1
    ens_dir = 1 if p_ens > 0.5 else -1
    direction_match = synth_dir * ens_dir  # +1 or -1

    # Combined confidence (geometric mean of distances from 0.5)
    synth_conf = abs(p_synth - 0.5) * 2
    ens_conf = abs(p_ens - 0.5) * 2
    combined_conf = (synth_conf * ens_conf) ** 0.5

    return direction_match * combined_conf


def compute_edge_quality(
    agreement_score: float,
    confidence: float,
    vol_ratio: float = 1.0,
) -> EdgeQuality:
    """
    Determine overall edge quality from agreement, confidence, and volatility.

    Args:
        agreement_score: -1 to 1 from _agreement_score()
        confidence: 0-1 blended confidence
        vol_ratio: forward_vol / realized_vol (>1 = expanding)

    Returns:
        HIGH, MEDIUM, or LOW quality assessment
    """
    # Base score from agreement and confidence
    score = agreement_score * 0.5 + confidence * 0.5

    # Volatility adjustment: expanding vol with agreement = better edge opportunity
    if vol_ratio > 1.2 and agreement_score > 0.3:
        score += 0.1  # expanding vol + agreement = bonus
    elif vol_ratio < 0.8 and agreement_score > 0.3:
        score -= 0.05  # contracting vol = slightly less opportunity

    if score > 0.4:
        return EdgeQuality.HIGH
    elif score > 0.15:
        return EdgeQuality.MEDIUM
    else:
        return EdgeQuality.LOW


def get_three_way_comparison(
    synthdata_prob_up: float,
    ensemble_prob_up: float,
    polymarket_prob_up: float,
) -> dict:
    """
    Compare all three signal sources and determine unanimous agreement.

    Args:
        synthdata_prob_up: P(UP) from SynthData
        ensemble_prob_up: P(UP) from local ensemble
        polymarket_prob_up: P(UP) from Polymarket

    Returns:
        {
            "synth_dir": "UP"|"DOWN",
            "ensemble_dir": "UP"|"DOWN",
            "polymarket_dir": "UP"|"DOWN",
            "all_agree": bool,
            "agree_direction": "UP"|"DOWN"|None,
            "sources_up": int,  # count of sources saying UP
            "conviction": float,  # 0-1, higher when all agree strongly
        }
    """
    synth_dir = "UP" if synthdata_prob_up > 0.5 else "DOWN"
    ens_dir = "UP" if ensemble_prob_up > 0.5 else "DOWN"
    poly_dir = "UP" if polymarket_prob_up > 0.5 else "DOWN"

    sources_up = sum(1 for d in [synth_dir, ens_dir, poly_dir] if d == "UP")
    all_agree = sources_up == 3 or sources_up == 0
    agree_direction = None
    if sources_up == 3:
        agree_direction = "UP"
    elif sources_up == 0:
        agree_direction = "DOWN"

    # Conviction: average distance from 0.5 when all agree, lower otherwise
    distances = [
        abs(synthdata_prob_up - 0.5),
        abs(ensemble_prob_up - 0.5),
        abs(polymarket_prob_up - 0.5),
    ]
    avg_distance = sum(distances) / len(distances)
    conviction = avg_distance * 2 * (1.0 if all_agree else 0.5)

    return {
        "synth_dir": synth_dir,
        "ensemble_dir": ens_dir,
        "polymarket_dir": poly_dir,
        "all_agree": all_agree,
        "agree_direction": agree_direction,
        "sources_up": sources_up,
        "conviction": min(1.0, conviction),
    }


if __name__ == "__main__":
    # Test with mock data
    print("=== Signal Blender Tests ===\n")

    # Test 1: Strong agreement UP
    result = blend_predictions(0.65, 0.62)
    print(f"Both UP (65%, 62%): blended={result['blended_prob_up']:.3f} "
          f"{result['blended_direction']} agreement={result['agreement'].value} "
          f"quality={result['quality'].value}")

    # Test 2: Strong agreement DOWN
    result = blend_predictions(0.35, 0.38)
    print(f"Both DOWN (35%, 38%): blended={result['blended_prob_up']:.3f} "
          f"{result['blended_direction']} agreement={result['agreement'].value} "
          f"quality={result['quality'].value}")

    # Test 3: Disagreement
    result = blend_predictions(0.65, 0.40)
    print(f"Disagree (65%, 40%): blended={result['blended_prob_up']:.3f} "
          f"{result['blended_direction']} agreement={result['agreement'].value} "
          f"quality={result['quality'].value}")

    # Test 4: Strong disagreement
    result = blend_predictions(0.72, 0.30)
    print(f"Strong disagree (72%, 30%): blended={result['blended_prob_up']:.3f} "
          f"{result['blended_direction']} agreement={result['agreement'].value} "
          f"quality={result['quality'].value}")

    # Test 5: 15min with only SynthData
    result = blend_synthdata_only(0.62, ensemble_direction_1h="UP")
    print(f"15min SynthData-only (62%, 1h=UP): blended={result['blended_prob_up']:.3f} "
          f"{result['blended_direction']} agreement={result['agreement'].value}")

    # Test 6: Neutral signals
    result = blend_predictions(0.51, 0.52)
    print(f"Neutral (51%, 52%): blended={result['blended_prob_up']:.3f} "
          f"{result['blended_direction']} agreement={result['agreement'].value} "
          f"quality={result['quality'].value}")
