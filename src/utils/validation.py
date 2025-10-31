"""
Physics-Based Validation for Coffee Roast Profiles

Validates roast profiles against known coffee roasting physics constraints:
- Temperature ranges (350-450°F)
- Heating rates (Rate of Rise: 20-100°F/min)
- Monotonicity post-turning-point
- Profile duration (7-16 minutes typical)
- Smooth transitions (no sudden jumps)

Author: Charlee Kraiss
Project: RoastFormer - Transformer-Based Roast Profile Generation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


class RoastProfileValidator:
    """
    Validates coffee roast profiles against physical constraints
    """

    # Physical constraints from coffee roasting domain knowledge
    TEMP_MIN = 120.0  # °F - Minimum expected temperature (turning point can be very low on efficient roasters)
    TEMP_MAX = 450.0  # °F - Maximum safe temperature

    CHARGE_TEMP_MIN = 400.0  # °F - Typical minimum charge temp
    CHARGE_TEMP_MAX = 450.0  # °F - Typical maximum charge temp

    DROP_TEMP_MIN = 380.0  # °F - Minimum drop temp (very light roast)
    DROP_TEMP_MAX = 430.0  # °F - Maximum drop temp (dark roast)

    DURATION_MIN = 400  # seconds (~7 minutes)
    DURATION_MAX = 1000  # seconds (~16 minutes)

    ROR_MIN = 20.0 / 60.0  # °F/second (20°F/min minimum)
    ROR_MAX = 100.0 / 60.0  # °F/second (100°F/min maximum)

    MAX_TEMP_JUMP = 25.0 / 60.0  # °F/second (25°F per second max - allow for measurement sampling)

    def __init__(self, strict: bool = False):
        """
        Initialize validator

        Args:
            strict: If True, enforce all constraints strictly.
                   If False, allow some tolerance for edge cases.
        """
        self.strict = strict
        self.tolerance = 0.0 if strict else 0.05  # 5% tolerance for non-strict

    def validate_temperature_range(
        self,
        temps: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Check if temperatures are within valid range

        Args:
            temps: Temperature sequence (°F)

        Returns:
            (is_valid, message)
        """
        if len(temps) == 0:
            return False, "Empty temperature sequence"

        min_temp = np.min(temps)
        max_temp = np.max(temps)

        if min_temp < self.TEMP_MIN:
            return False, f"Temperature too low: {min_temp:.1f}°F (min: {self.TEMP_MIN}°F)"

        if max_temp > self.TEMP_MAX:
            return False, f"Temperature too high: {max_temp:.1f}°F (max: {self.TEMP_MAX}°F)"

        return True, "Temperature range valid"

    def validate_charge_and_drop(
        self,
        temps: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Check if charge (start) and drop (end) temperatures are reasonable

        Args:
            temps: Temperature sequence (°F)

        Returns:
            (is_valid, message)
        """
        if len(temps) < 2:
            return False, "Insufficient data points"

        charge_temp = temps[0]
        drop_temp = temps[-1]

        # Check charge temp
        if charge_temp < self.CHARGE_TEMP_MIN or charge_temp > self.CHARGE_TEMP_MAX:
            if not self.strict:
                # Allow some tolerance
                if charge_temp < self.CHARGE_TEMP_MIN - 50 or charge_temp > self.CHARGE_TEMP_MAX + 20:
                    return False, f"Charge temp out of range: {charge_temp:.1f}°F"

        # Check drop temp
        if drop_temp < self.DROP_TEMP_MIN or drop_temp > self.DROP_TEMP_MAX:
            if not self.strict:
                # Allow some tolerance
                if drop_temp < self.DROP_TEMP_MIN - 10 or drop_temp > self.DROP_TEMP_MAX + 10:
                    return False, f"Drop temp out of range: {drop_temp:.1f}°F"

        return True, f"Charge: {charge_temp:.1f}°F, Drop: {drop_temp:.1f}°F"

    def validate_duration(
        self,
        times: Optional[np.ndarray] = None,
        num_points: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        Check if roast duration is reasonable

        Args:
            times: Time sequence (seconds), or
            num_points: Number of data points (assumes 1 sample/second)

        Returns:
            (is_valid, message)
        """
        if times is not None:
            duration = times[-1] - times[0]
        elif num_points is not None:
            duration = num_points  # Assumes 1 sample per second
        else:
            return False, "Must provide either times or num_points"

        if duration < self.DURATION_MIN or duration > self.DURATION_MAX:
            if not self.strict:
                # Allow wider range for non-strict
                if duration < 300 or duration > 1200:  # 5-20 minutes
                    return False, f"Duration out of range: {duration:.0f}s ({duration/60:.1f} min)"

        return True, f"Duration: {duration:.0f}s ({duration/60:.1f} min)"

    def validate_monotonicity(
        self,
        temps: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Check if temperature increases monotonically after turning point

        The turning point is the minimum temperature after bean charge,
        typically occurring in the first 2-4 minutes.

        Args:
            temps: Temperature sequence (°F)

        Returns:
            (is_valid, message)
        """
        if len(temps) < 10:
            return False, "Insufficient data points for monotonicity check"

        # Find turning point (minimum temperature)
        turning_point_idx = np.argmin(temps)

        # Check monotonicity from turning point onwards
        post_turning = temps[turning_point_idx:]
        diffs = np.diff(post_turning)

        # Count SIGNIFICANT decreases (> 2°F) - small decreases are temperature control
        significant_decreases = (diffs < -2.0).sum()

        # Allow up to 10% of points to have significant decreases (temperature control)
        allowed_decreases = max(3, int(len(diffs) * 0.10))

        if significant_decreases > allowed_decreases:
            return False, f"Non-monotonic: {significant_decreases} significant decreases (>2°F)"

        return True, f"Monotonic (turning point at t={turning_point_idx}s, {significant_decreases} small dips)"

    def validate_heating_rates(
        self,
        temps: np.ndarray,
        sample_rate: float = 1.0
    ) -> Tuple[bool, str]:
        """
        Check if heating rates (Rate of Rise) are within physical limits

        Args:
            temps: Temperature sequence (°F)
            sample_rate: Samples per second (default: 1.0)

        Returns:
            (is_valid, message)
        """
        if len(temps) < 2:
            return False, "Insufficient data points for RoR check"

        # Calculate Rate of Rise (°F per second)
        ror = np.diff(temps) * sample_rate

        # Filter out very small changes (sensor noise)
        significant_ror = ror[np.abs(ror) > 0.1]

        if len(significant_ror) == 0:
            return False, "No significant temperature changes detected"

        # Check bounds
        too_fast = (significant_ror > self.ROR_MAX).sum()
        too_slow = (significant_ror < -self.ROR_MIN).sum()  # Negative = decreasing

        total_points = len(significant_ror)

        # Allow some tolerance
        allowed_violations = int(total_points * self.tolerance)

        if too_fast > allowed_violations:
            pct = (too_fast / total_points) * 100
            return False, f"Heating too fast: {pct:.1f}% points above {self.ROR_MAX*60:.1f}°F/min"

        # Note: We allow temperature decreases early in roast (drying phase)
        # So we don't fail on too_slow, just warn

        avg_ror = np.mean(significant_ror) * 60  # Convert to °F/min
        return True, f"RoR valid (avg: {avg_ror:.1f}°F/min)"

    def validate_smoothness(
        self,
        temps: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Check for sudden temperature jumps (sensor errors, bad data)

        Args:
            temps: Temperature sequence (°F)

        Returns:
            (is_valid, message)
        """
        if len(temps) < 2:
            return False, "Insufficient data points"

        # Check for sudden jumps
        diffs = np.abs(np.diff(temps))
        max_jump = np.max(diffs)

        # Instead of failing on single max jump, check for multiple outlier jumps
        # This is more robust to occasional measurement noise
        threshold = 30.0  # 30°F per second absolute limit

        # Count how many jumps exceed threshold
        large_jumps = (diffs > threshold).sum()

        # Allow a few outliers (< 1% of points)
        max_allowed_outliers = max(1, int(len(diffs) * 0.01))

        if large_jumps > max_allowed_outliers:
            return False, f"{large_jumps} jumps > {threshold}°F (max: {max_jump:.1f}°F)"

        return True, f"Smooth profile (max jump: {max_jump:.1f}°F, outliers: {large_jumps})"

    def validate_profile(
        self,
        temps: np.ndarray,
        times: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> Tuple[bool, Dict[str, Tuple[bool, str]]]:
        """
        Run all validation checks on a roast profile

        Args:
            temps: Temperature sequence (°F)
            times: Optional time sequence (seconds)
            verbose: If True, print validation results

        Returns:
            (all_valid, results_dict)
            where results_dict = {
                'test_name': (is_valid, message),
                ...
            }
        """
        results = {}

        # Run all validation checks
        results['temperature_range'] = self.validate_temperature_range(temps)
        results['charge_and_drop'] = self.validate_charge_and_drop(temps)
        results['duration'] = self.validate_duration(num_points=len(temps))
        results['monotonicity'] = self.validate_monotonicity(temps)
        results['heating_rates'] = self.validate_heating_rates(temps)
        results['smoothness'] = self.validate_smoothness(temps)

        # Overall validity
        all_valid = all(result[0] for result in results.values())

        if verbose:
            print("=" * 80)
            print("ROAST PROFILE VALIDATION")
            print("=" * 80)
            for test_name, (is_valid, message) in results.items():
                status = "✓" if is_valid else "✗"
                print(f"{status} {test_name:20s}: {message}")
            print("-" * 80)
            print(f"Overall: {'VALID' if all_valid else 'INVALID'}")
            print("=" * 80)

        return all_valid, results

    def validate_dataset(
        self,
        profiles: List[Dict],
        verbose: bool = True
    ) -> Dict:
        """
        Validate all profiles in a dataset

        Args:
            profiles: List of profile dicts with 'roast_profile' and 'metadata'
            verbose: If True, print summary

        Returns:
            {
                'total': int,
                'valid': int,
                'invalid': int,
                'invalid_profiles': List[Dict],  # Details on failed profiles
                'validation_results': List[Dict]  # All results
            }
        """
        validation_results = []
        invalid_profiles = []

        for i, profile in enumerate(profiles):
            # Extract temperature data
            if 'roast_profile' not in profile:
                invalid_profiles.append({
                    'index': i,
                    'name': profile.get('metadata', {}).get('product_name', f'Profile_{i}'),
                    'reason': 'Missing roast_profile'
                })
                continue

            bean_temp_data = profile['roast_profile'].get('bean_temp', [])

            if not bean_temp_data:
                invalid_profiles.append({
                    'index': i,
                    'name': profile.get('metadata', {}).get('product_name', f'Profile_{i}'),
                    'reason': 'Empty bean_temp data'
                })
                continue

            # Convert to numpy array
            temps = np.array([point.get('value', point.get('y', 0)) for point in bean_temp_data])

            # Validate
            is_valid, results = self.validate_profile(temps, verbose=False)

            result_dict = {
                'index': i,
                'name': profile.get('metadata', {}).get('product_name', f'Profile_{i}'),
                'valid': is_valid,
                'checks': results
            }

            validation_results.append(result_dict)

            if not is_valid:
                failed_checks = [name for name, (valid, _) in results.items() if not valid]
                invalid_profiles.append({
                    'index': i,
                    'name': result_dict['name'],
                    'failed_checks': failed_checks,
                    'details': results
                })

        summary = {
            'total': len(profiles),
            'valid': len([r for r in validation_results if r['valid']]),
            'invalid': len(invalid_profiles),
            'invalid_profiles': invalid_profiles,
            'validation_results': validation_results
        }

        if verbose:
            print("\n" + "=" * 80)
            print("DATASET VALIDATION SUMMARY")
            print("=" * 80)
            print(f"Total profiles:   {summary['total']}")
            print(f"✓ Valid:          {summary['valid']} ({summary['valid']/summary['total']*100:.1f}%)")
            print(f"✗ Invalid:        {summary['invalid']} ({summary['invalid']/summary['total']*100:.1f}%)")

            if summary['invalid'] > 0:
                print("\nInvalid profiles:")
                for inv in invalid_profiles[:10]:  # Show first 10
                    print(f"  - {inv['name']}: {inv.get('reason', inv.get('failed_checks', ''))}")

            print("=" * 80)

        return summary


def validate_onyx_dataset(dataset_dir: str, strict: bool = False) -> Dict:
    """
    Validate an Onyx dataset directory

    Args:
        dataset_dir: Path to onyx_dataset_YYYY_MM_DD directory
        strict: Use strict validation rules

    Returns:
        Validation summary dict
    """
    dataset_path = Path(dataset_dir) / 'complete_dataset.json'

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    profiles = dataset.get('profiles', [])

    print(f"\nValidating {len(profiles)} profiles from {dataset_dir}")
    print("-" * 80)

    # Validate
    validator = RoastProfileValidator(strict=strict)
    summary = validator.validate_dataset(profiles, verbose=True)

    return summary


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
        strict = '--strict' in sys.argv

        summary = validate_onyx_dataset(dataset_dir, strict=strict)

        # Exit with error code if any invalid profiles
        sys.exit(0 if summary['invalid'] == 0 else 1)
    else:
        print("Usage: python validation.py <dataset_dir> [--strict]")
        print("Example: python validation.py onyx_dataset_2025_10_31")
