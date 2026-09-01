import re
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pydicom
import SimpleITK as sitk

from ..exceptions import DataStructureError, DataStructureWarning

ENHANCED_PET_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.130"


def _is_enhanced_pet(ds):
    return str(getattr(ds, "SOPClassUID", "")) == ENHANCED_PET_SOP_CLASS_UID


def reject_unsupported_enhanced_pet(dicom_files):
    if any(_is_enhanced_pet(dicom_file["ds"]) for dicom_file in dicom_files):
        raise DataStructureError("Enhanced PET Image Storage is not supported.")


def is_fdg(name):
    fdg_pattern = re.compile(
        r"(fdg|fluorodeoxy|fludeoxy|2[-\s]?\[?18f\]?[-\s]?fluoro)",
        re.IGNORECASE,
    )
    return bool(fdg_pattern.search(name))


def parse_time(time_str, vr=None):
    """Parse a DICOM DA, TM, or DT value into a datetime object."""
    if isinstance(time_str, bytes):
        time_str = time_str.decode("utf-8").strip()

    time_str = str(time_str).strip()
    match = re.fullmatch(r"(?P<components>\d+)(?P<fraction>\.\d{1,6})?(?P<offset>[+-]\d{4})?", time_str)
    if match is None:
        raise ValueError(f"Time data '{time_str}' does not match a DICOM date or time format")

    components = match.group("components")
    fraction = match.group("fraction") or ""
    offset = match.group("offset") or ""
    vr = vr.upper() if vr is not None else None
    if vr is None:
        if offset or len(components) > 8:
            vr = "DT"
        elif len(components) == 8:
            vr = "DA"
        else:
            vr = "TM"

    formats = {
        "DA": {8: "%Y%m%d"},
        "TM": {2: "%H", 4: "%H%M", 6: "%H%M%S"},
        "DT": {
            4: "%Y",
            6: "%Y%m",
            8: "%Y%m%d",
            10: "%Y%m%d%H",
            12: "%Y%m%d%H%M",
            14: "%Y%m%d%H%M%S",
        },
    }
    if vr not in formats or len(components) not in formats[vr]:
        raise ValueError(f"Time data '{time_str}' has invalid component width for DICOM {vr or 'value'}")
    if fraction and not ((vr == "TM" and len(components) == 6) or (vr == "DT" and len(components) == 14)):
        raise ValueError(f"Time data '{time_str}' has fractional seconds without a seconds component")
    if offset and vr != "DT":
        raise ValueError(f"Time data '{time_str}' has a UTC offset but is not a DICOM DT value")

    fmt = formats[vr][len(components)]
    if fraction:
        fmt += ".%f"
    if offset:
        fmt += "%z"
    return datetime.strptime(components + fraction + offset, fmt)


def _dataset_timezone(ds):
    dataset_offset = getattr(ds, "TimezoneOffsetFromUTC", None)
    if dataset_offset in [None, ""]:
        return None
    match = re.fullmatch(
        r"(?P<sign>[+-])(?P<hours>\d{2})(?P<minutes>\d{2})",
        str(dataset_offset).strip(),
    )
    if match is None:
        raise DataStructureError("Timezone Offset From UTC (0008,0201) is invalid.")
    minutes = int(match.group("minutes"))
    total_minutes = int(match.group("hours")) * 60 + minutes
    if (
        minutes >= 60
        or (match.group("sign") == "+" and total_minutes > 14 * 60)
        or (match.group("sign") == "-" and total_minutes > 12 * 60)
    ):
        raise DataStructureError("Timezone Offset From UTC (0008,0201) is invalid.")
    if match.group("sign") == "-":
        total_minutes = -total_minutes
    return timezone(timedelta(minutes=total_minutes))


def _datetime_on_reference_date(value, reference, infer_previous_day=True):
    # Vendor-private reference fields may contain either TM or a complete DT.
    parsed = parse_time(value)
    tzinfo = parsed.tzinfo if parsed.tzinfo is not None else reference.tzinfo
    raw_value = value.decode("utf-8").strip() if isinstance(value, bytes) else str(value).strip()
    components = re.match(r"\d+", raw_value)
    if components is not None and len(components.group()) >= 8:
        return parsed.replace(tzinfo=tzinfo)

    result = parsed.replace(
        year=reference.year,
        month=reference.month,
        day=reference.day,
        tzinfo=tzinfo,
    )
    if infer_previous_day and result > reference:
        result -= timedelta(days=1)
    return result


def get_radionuclide_half_life(ds):
    """Return radionuclide half-life in seconds."""
    rph = ds.RadiopharmaceuticalInformationSequence[0]

    try:
        half_life = float(rph.RadionuclideHalfLife)
    except (AttributeError, TypeError, ValueError):
        raise DataStructureError("Radionuclide Half Life (0018,1075) is missing or invalid.")

    if half_life <= 0:
        raise DataStructureError("Radionuclide Half Life (0018,1075) must be > 0.")

    return half_life


def get_decay_correction_reference_datetime(ds, acquisition_time, decay_constant):
    """
    Return the datetime to which the PET activity is decay-corrected.

    This mirrors the existing vendor-specific START decay-correction logic and
    is used only to determine whether RadiopharmaceuticalStartDateTime has a
    plausible date offset.
    """
    if ds.DecayCorrection != "START":
        return acquisition_time

    manufacturer = ds.Manufacturer.upper()

    series_time = None
    try:
        series_date_value = getattr(ds, "SeriesDate", None)
        if series_date_value not in [None, ""]:
            series_date = parse_time(series_date_value, "DA")
            series_time = _datetime_on_reference_date(ds.SeriesTime, series_date, infer_previous_day=False)
        else:
            series_time = _datetime_on_reference_date(ds.SeriesTime, acquisition_time)
        if series_time.tzinfo is None and acquisition_time.tzinfo is not None:
            series_time = series_time.replace(tzinfo=acquisition_time.tzinfo)
    except (AttributeError, ValueError, TypeError):
        pass

    if series_time is not None and acquisition_time == series_time:
        return acquisition_time

    if "SIEMENS" in manufacturer or "CPS" in manufacturer or "CTI" in manufacturer:
        try:
            return _datetime_on_reference_date(ds[(0x0071, 0x1022)].value, acquisition_time)
        except (KeyError, TypeError, ValueError):
            pass

    if "GE" in manufacturer:
        try:
            return _datetime_on_reference_date(ds[(0x0009, 0x100D)].value, acquisition_time)
        except (KeyError, TypeError, ValueError):
            frame_reference_time = float(ds.FrameReferenceTime) / 1000.0
            return acquisition_time - timedelta(seconds=frame_reference_time)

    frame_reference_time = float(ds.FrameReferenceTime) / 1000.0
    decay_during_frame = decay_constant * float(ds.ActualFrameDuration) / 1000.0
    avg_count_rate_time = (1 / decay_constant) * np.log(decay_during_frame / (1 - np.exp(-decay_during_frame)))

    return acquisition_time + timedelta(seconds=avg_count_rate_time - frame_reference_time)


def resolve_injection_and_acquisition_times(ds, half_life, decay_constant):
    """
    Resolve injection and acquisition datetimes.

    A complete RadiopharmaceuticalStartDateTime is trusted only when its offset
    to the decay-correction reference datetime is >= -1 hour and < 2 half-lives.
    For uncorrected frames, the reference is the acquisition datetime itself,
    so the offset must be nonnegative.

    For short-lived radionuclides, unreliable or missing dates can be replaced
    by a time-based inference. If injection time is later than acquisition time,
    injection is assumed to have occurred on the previous day, but only when
    the resulting difference is < 6 hours.

    For long-lived radionuclides, dates must be trustworthy because uptake may
    legitimately span more than one day.
    """
    rph = ds.RadiopharmaceuticalInformationSequence[0]

    injection_datetime_value = getattr(rph, "RadiopharmaceuticalStartDateTime", None)
    injection_time_value = getattr(rph, "RadiopharmaceuticalStartTime", None)
    acquisition_date_value = getattr(ds, "AcquisitionDate", None)

    injection_datetime_present = injection_datetime_value not in [None, ""]
    injection_time_present = injection_time_value not in [None, ""]
    acquisition_date_present = acquisition_date_value not in [None, ""]

    if not injection_datetime_present and not injection_time_present:
        raise DataStructureError(
            "Both Radiopharmaceutical Start DateTime (0018,1078) and "
            "Radiopharmaceutical Start Time (0018,1072) are missing."
        )

    injection_datetime = None
    if injection_datetime_present:
        try:
            injection_datetime = parse_time(injection_datetime_value, "DT")
        except (ValueError, TypeError):
            raise DataStructureError("Radiopharmaceutical Start DateTime (0018,1078) is invalid.")

    if injection_time_present:
        injection_clock = parse_time(injection_time_value, "TM")
    else:
        injection_clock = injection_datetime

    acquisition_clock = parse_time(ds.AcquisitionTime, "TM")
    dataset_timezone = _dataset_timezone(ds)
    if injection_datetime is not None and injection_datetime.tzinfo is None and dataset_timezone is not None:
        injection_datetime = injection_datetime.replace(tzinfo=dataset_timezone)
    if injection_clock.tzinfo is None:
        injection_timezone = dataset_timezone
        if injection_timezone is None and injection_datetime is not None:
            injection_timezone = injection_datetime.tzinfo
        if injection_timezone is not None:
            injection_clock = injection_clock.replace(tzinfo=injection_timezone)
    if acquisition_clock.tzinfo is None:
        acquisition_timezone = dataset_timezone
        if acquisition_timezone is None and injection_datetime is not None:
            # Preserve support for legacy datasets containing only one explicit
            # offset on the administration DT.
            acquisition_timezone = injection_datetime.tzinfo
        if acquisition_timezone is not None:
            acquisition_clock = acquisition_clock.replace(tzinfo=acquisition_timezone)

    # According to the DRO recommendations, long-lived radionuclides require
    # a reliable full administration datetime because uptake can exceed 24 h.
    long_lived_radionuclide = half_life >= 41400
    minimum_reference_offset = 0 if ds.DecayCorrection == "NONE" else -3600

    def validate_reconstructed_times(injection_time, acquisition_time):
        decay_reference_time = get_decay_correction_reference_datetime(
            ds,
            acquisition_time,
            decay_constant,
        )
        reconstructed_offset = (decay_reference_time - injection_time).total_seconds()
        if not minimum_reference_offset <= reconstructed_offset < 2 * half_life:
            raise DataStructureError(
                "Reconstructed administration and acquisition times are inconsistent "
                "with the decay-correction reference datetime."
            )
        return injection_time, acquisition_time

    if not injection_datetime_present and long_lived_radionuclide:
        raise DataStructureError(
            "Radiopharmaceutical Start DateTime (0018,1078) is required "
            "for a long-lived radionuclide because the uptake time may span "
            "more than one day."
        )

    if acquisition_date_present:
        acquisition_date = parse_time(acquisition_date_value, "DA")
        acquisition_time = acquisition_clock.replace(
            year=acquisition_date.year,
            month=acquisition_date.month,
            day=acquisition_date.day,
        )
    else:
        if long_lived_radionuclide:
            raise DataStructureError(
                "Acquisition Date is missing for a long-lived radionuclide. "
                "The uptake time cannot be determined reliably."
            )

        if injection_datetime is not None:
            acquisition_time = acquisition_clock.replace(
                year=injection_datetime.year,
                month=injection_datetime.month,
                day=injection_datetime.day,
            )
        else:
            acquisition_time = acquisition_clock

    # If both complete dates are present, determine whether they are mutually
    # plausible using the actual decay-correction reference datetime.
    if injection_datetime is not None and acquisition_date_present:
        decay_reference_time = get_decay_correction_reference_datetime(
            ds,
            acquisition_time,
            decay_constant,
        )

        datetime_offset = (decay_reference_time - injection_datetime).total_seconds()

        if minimum_reference_offset <= datetime_offset < 2 * half_life:
            return injection_datetime, acquisition_time

        if ds.DecayCorrection == "NONE" and datetime_offset < 0:
            raise DataStructureError(
                "Radiopharmaceutical Start DateTime is after the acquisition datetime for an uncorrected PET frame."
            )

        # An implausible full datetime for a long-lived radionuclide cannot
        # safely be repaired from clock times (DRO_error_4_2).
        if long_lived_radionuclide:
            raise DataStructureError(
                "Radiopharmaceutical Start DateTime is inconsistent with the "
                "decay-correction reference datetime for a long-lived "
                "radionuclide. The administration date may have been anonymized "
                "or shifted and cannot be inferred safely."
            )

    # From here on, the radionuclide is short-lived and either one date is
    # missing or the complete datetime offset was implausible. Ignore the
    # unreliable date offset and reconstruct only the relative day relationship.
    if injection_datetime is not None:
        injection_time = injection_clock.replace(
            year=injection_datetime.year,
            month=injection_datetime.month,
            day=injection_datetime.day,
        )

        acquisition_time = acquisition_clock.replace(
            year=injection_datetime.year,
            month=injection_datetime.month,
            day=injection_datetime.day,
        )

        if injection_time.time() > acquisition_time.time():
            acquisition_time += timedelta(days=1)

            if (acquisition_time - injection_time).total_seconds() >= 6 * 3600:
                raise DataStructureError(
                    "Injection time is later than acquisition time, but "
                    "assuming a midnight rollover results in a time difference "
                    "of 6 hours or more."
                )

        return validate_reconstructed_times(injection_time, acquisition_time)

    if acquisition_date_present:
        acquisition_date = parse_time(acquisition_date_value, "DA")

        acquisition_time = acquisition_clock.replace(
            year=acquisition_date.year,
            month=acquisition_date.month,
            day=acquisition_date.day,
        )

        injection_time = injection_clock.replace(
            year=acquisition_date.year,
            month=acquisition_date.month,
            day=acquisition_date.day,
        )

        if injection_time.time() > acquisition_time.time():
            injection_time -= timedelta(days=1)

            if (acquisition_time - injection_time).total_seconds() >= 6 * 3600:
                raise DataStructureError(
                    "Injection time is later than acquisition time while the "
                    "injection date is missing, but assuming injection on the "
                    "previous day results in a time difference of 6 hours or more."
                )

        return validate_reconstructed_times(injection_time, acquisition_time)

    injection_time = injection_clock
    acquisition_time = acquisition_clock

    if injection_time.time() > acquisition_time.time():
        injection_time -= timedelta(days=1)

        if (acquisition_time - injection_time).total_seconds() >= 6 * 3600:
            raise DataStructureError(
                "Injection time is later than acquisition time while both dates "
                "are missing, but assuming injection on the previous day results "
                "in a time difference of 6 hours or more."
            )

    return validate_reconstructed_times(injection_time, acquisition_time)


def calc_elapsed_time(ds, decay_constant, acquisition_time, injection_time):
    frame_reference_time = float(ds.FrameReferenceTime) / 1000
    decay_during_frame = decay_constant * ds.ActualFrameDuration / 1000
    avg_count_rate_time = (1 / decay_constant) * np.log(decay_during_frame / (1 - np.exp(-decay_during_frame)))

    return (acquisition_time - injection_time).total_seconds() + avg_count_rate_time - frame_reference_time


def get_patient_height_cm(ds):
    """Return patient height in cm."""
    if hasattr(ds, "PatientSize") and ds.PatientSize not in [None, ""]:
        height = float(ds.PatientSize)
    elif (0x0010, 0x1020) in ds and ds[(0x0010, 0x1020)].value not in [None, ""]:
        height = float(ds[(0x0010, 0x1020)].value)
    else:
        raise DataStructureError("Patient height tag (0010,1020) is missing.")

    if height <= 0:
        raise DataStructureError("Patient height must be > 0.")

    return height * 100.0 if height <= 3 else height


def calculate_bsa_du_bois(height_cm, weight_kg):
    """Calculate Du Bois body surface area in m^2."""
    if height_cm <= 0 or weight_kg <= 0:
        raise DataStructureError("Height and weight must be > 0 to compute BSA.")
    return 0.007184 * (height_cm**0.725) * (weight_kg**0.425)


def get_patient_sex(ds):
    """Return normalized patient sex: 'M', 'F', or 'O'."""
    sex = getattr(ds, "PatientSex", None)
    if sex is None and (0x0010, 0x0040) in ds:
        sex = ds[(0x0010, 0x0040)].value

    if sex is None:
        raise DataStructureError("Patient sex tag (0010,0040) is missing.")

    sex = str(sex).strip().upper()
    if sex == "":
        raise DataStructureError("Patient sex tag (0010,0040) is empty.")
    if sex not in {"M", "F", "O"}:
        raise DataStructureError(f"Unsupported PatientSex '{sex}'. Expected one of 'M', 'F', or 'O'.")
    return sex


def calculate_lbm_morgan(height_cm, weight_kg, sex):
    """Calculate lean body mass using the Morgan/Sugawara-style formula."""
    if height_cm <= 0 or weight_kg <= 0:
        raise DataStructureError("Height and weight must be > 0 to compute LBM.")

    male_lbm = 1.10 * weight_kg - 120.0 * ((weight_kg / height_cm) ** 2)
    female_lbm = 1.07 * weight_kg - 148.0 * ((weight_kg / height_cm) ** 2)

    if sex == "M":
        lbm = male_lbm
    elif sex == "F":
        lbm = female_lbm
    elif sex == "O":
        lbm = 0.5 * (male_lbm + female_lbm)
    else:
        raise DataStructureError(f"Unsupported sex '{sex}' for LBM calculation.")

    if lbm <= 0:
        raise DataStructureError(f"Computed Morgan LBM is non-positive: {lbm}.")
    return lbm


def calculate_lbm_james128(height_cm, weight_kg, sex):
    """Calculate lean body mass using the James/Morgan-128 formula."""
    if height_cm <= 0 or weight_kg <= 0:
        raise DataStructureError("Height and weight must be > 0 to compute LBM.")

    male_lbm = 1.10 * weight_kg - 128.0 * ((weight_kg / height_cm) ** 2)
    female_lbm = 1.07 * weight_kg - 148.0 * ((weight_kg / height_cm) ** 2)

    if sex == "M":
        lbm = male_lbm
    elif sex == "F":
        lbm = female_lbm
    elif sex == "O":
        lbm = 0.5 * (male_lbm + female_lbm)
    else:
        raise DataStructureError(f"Unsupported sex '{sex}' for LBM calculation.")

    if lbm <= 0:
        raise DataStructureError(f"Computed James128 LBM is non-positive: {lbm}.")
    return lbm


def calculate_lbm_janmahasatian(height_cm, weight_kg, sex):
    """Calculate lean body mass using the Janmahasatian formula."""
    if height_cm <= 0 or weight_kg <= 0:
        raise DataStructureError("Height and weight must be > 0 to compute LBM.")

    height_m = height_cm * 1e-2
    bmi = weight_kg / (height_m**2)

    male_lbm = 9270.0 * weight_kg / (6680.0 + 216.0 * bmi)
    female_lbm = 9270.0 * weight_kg / (8780.0 + 244.0 * bmi)

    if sex == "M":
        lbm = male_lbm
    elif sex == "F":
        lbm = female_lbm
    elif sex == "O":
        lbm = 0.5 * (male_lbm + female_lbm)
    else:
        raise DataStructureError(f"Unsupported sex '{sex}' for LBM calculation.")

    if lbm <= 0:
        raise DataStructureError(f"Computed Janmahasatian LBM is non-positive: {lbm}.")
    return lbm


def calculate_ibw(height_cm, sex):
    """Calculate ideal body weight."""
    if height_cm <= 0:
        raise DataStructureError("Height must be > 0 to compute IBW.")

    male_ibw = 48.0 + 1.06 * (height_cm - 152.0)
    female_ibw = 45.5 + 0.91 * (height_cm - 152.0)

    if sex == "M":
        ibw = male_ibw
    elif sex == "F":
        ibw = female_ibw
    elif sex == "O":
        ibw = 0.5 * (male_ibw + female_ibw)
    else:
        raise DataStructureError(f"Unsupported sex '{sex}' for IBW calculation.")

    if ibw <= 0:
        raise DataStructureError(f"Computed IBW is non-positive: {ibw}.")
    return ibw


def _gml_suv_type(ds):
    suv_type_elem = ds.get((0x0054, 0x1006), None)
    suv_type = (
        "BW" if suv_type_elem is None or suv_type_elem.value in [None, ""] else str(suv_type_elem.value).strip().upper()
    )
    supported_types = {"BW", "LBM", "LBMJAMES128", "LBMJANMA", "IBW"}
    if suv_type not in supported_types:
        raise DataStructureError(
            f"GML with SUV Type '{suv_type}' is not supported. "
            f"Supported types are BW, LBM, LBMJAMES128, LBMJANMA, and IBW."
        )
    return suv_type


def get_gml_normalization_info(ds):
    """Parse GML SUV normalization metadata and compute the normalization factor."""
    suv_type = _gml_suv_type(ds)

    try:
        patient_weight = float(ds.PatientWeight)
    except Exception:
        raise DataStructureError("Patient weight tag is missing or invalid for GML normalization.")

    if patient_weight <= 0:
        raise DataStructureError("Patient weight must be > 0 for GML normalization.")

    if suv_type == "BW":
        return suv_type, patient_weight

    height_cm = get_patient_height_cm(ds)
    sex = get_patient_sex(ds)

    if suv_type == "LBM":
        factor = calculate_lbm_morgan(height_cm, patient_weight, sex)
    elif suv_type == "LBMJAMES128":
        factor = calculate_lbm_james128(height_cm, patient_weight, sex)
    elif suv_type == "LBMJANMA":
        factor = calculate_lbm_janmahasatian(height_cm, patient_weight, sex)
    elif suv_type == "IBW":
        factor = calculate_ibw(height_cm, sex)

    return suv_type, factor


def validate_pet_dicom_tags(dicom_files):
    reject_unsupported_enhanced_pet(dicom_files)

    for dcm_file in dicom_files:
        ds = dcm_file["ds"]
        image_id = dcm_file["file_path"]

        try:
            pat_weight = ds[(0x0010, 0x1030)].value
            if float(pat_weight) < 1:
                warning_msg = f"For patient's {image_id} image, patient's weight tag (0071, 1022) contains weight < 1kg. Patient is excluded from the analysis."
                warnings.warn(warning_msg, DataStructureWarning)

        except (KeyError, TypeError):
            warning_msg = f"For patient's {image_id} image, patient's weight tag (0071, 1022) is not present. Patient is excluded from the analysis."
            warnings.warn(warning_msg, DataStructureWarning)
        if pat_weight <= 0:
            raise DataStructureError("Patient weight must be > 0.")
        if "DECY" not in ds[(0x0028, 0x0051)].value or "ATTN" not in ds[(0x0028, 0x0051)].value:
            warning_msg = f"For patient's {image_id} image, in DICOM tag (0028, 0051) either no 'DECY' (decay correction) or 'ATTN' (attenuation correction). Patient is excluded from the analysis."
            warnings.warn(warning_msg, DataStructureWarning)
        if ds.Units == "BQML":
            half_life = get_radionuclide_half_life(ds)
            decay_constant = np.log(2) / half_life

            if ds.DecayCorrection != "ADMIN":
                injection_time, acquisition_time = resolve_injection_and_acquisition_times(
                    ds,
                    half_life,
                    decay_constant,
                )

            if ds.DecayCorrection == "START":
                if "PHILIPS" in ds.Manufacturer.upper():
                    elapsed_time = calc_elapsed_time(
                        ds,
                        decay_constant,
                        acquisition_time,
                        injection_time,
                    )
                elif (
                    "SIEMENS" in ds.Manufacturer.upper()
                    or "CPS" in ds.Manufacturer.upper()
                    or "CTI" in ds.Manufacturer.upper()
                ):
                    try:
                        elapsed_time = (
                            _datetime_on_reference_date(ds[(0x0071, 0x1022)].value, acquisition_time) - injection_time
                        ).total_seconds()

                    except (KeyError, TypeError):
                        elapsed_time = calc_elapsed_time(
                            ds,
                            decay_constant,
                            acquisition_time,
                            injection_time,
                        )
                elif "GE" in ds.Manufacturer.upper():
                    try:
                        elapsed_time = (
                            _datetime_on_reference_date(ds[(0x0009, 0x100D)].value, acquisition_time) - injection_time
                        ).total_seconds()

                    except (KeyError, TypeError):
                        frame_reference_time = float(ds.FrameReferenceTime) / 1000

                        elapsed_time = (acquisition_time - injection_time).total_seconds() - frame_reference_time
                else:
                    elapsed_time = calc_elapsed_time(
                        ds,
                        decay_constant,
                        acquisition_time,
                        injection_time,
                    )

                    warning_msg = f"For patient's {image_id} image, an unknown PET scaner manufacturer is present {ds.Manufacturer}. Siemens/Philips strategy is applied."
                    warnings.warn(warning_msg, DataStructureWarning)

            elif ds.DecayCorrection == "ADMIN":
                elapsed_time = 0
            elif ds.DecayCorrection == "NONE":
                elapsed_time = None
            else:
                warning_msg = f"For patient's {image_id} image, An unsupported Decay Correction {ds.DecayCorrection} is present. Only supported are 'NONE', 'START' and 'ADMIN'. Patient is excluded from the analysis."
                raise DataStructureError(warning_msg)
            if elapsed_time is not None and elapsed_time < 0:
                error_msg = f"For patient's {image_id} image, patient is excluded from the analysis due to the negative time difference in the decay factor."
                raise DataStructureError(error_msg)
            elif (
                elapsed_time is not None
                and elapsed_time > 0
                and abs(elapsed_time) < 1800
                and ds.DecayCorrection != "ADMIN"
            ):
                warning_msg = f"Only {abs(elapsed_time) / 60} minutes after the injection."
                warnings.warn(warning_msg, DataStructureWarning)
        elif ds.Units == "CNTS" and "PHILIPS" in ds.Manufacturer.upper():
            if not (
                ((0x7053, 0x1009) in ds and ds[(0x7053, 0x1009)].value != 0)
                or ((0x7053, 0x1000) in ds and ds[(0x7053, 0x1000)].value != 0)
            ):
                error_msg = f"For patient's {image_id} image, patient is excluded, Philips scale factors not present (PET units CNTS)"
                raise DataStructureError(error_msg)
        elif ds.Units == "GML":
            try:
                _suv_type, _factor = get_gml_normalization_info(ds)
            except Exception as e:
                error_msg = f"For patient's {image_id} image, patient is excluded, GML normalization is invalid: {e}"
                raise DataStructureError(error_msg)

        elif ds.Units == "CM2ML":
            suv_type = ds.get((0x0054, 0x1006), None)
            if suv_type is not None and suv_type.value != "BSA":
                error_msg = f"For patient's {image_id} image, patient is excluded, SUV Type is not BSA (CM2ML units)"
                raise DataStructureError(error_msg)

            try:
                patient_weight = float(ds.PatientWeight)
                height_cm = get_patient_height_cm(ds)
                _ = calculate_bsa_du_bois(height_cm, patient_weight)
            except Exception as e:
                error_msg = (
                    f"For patient's {image_id} image, CM2ML requires valid patient "
                    f"height and weight for Du Bois BSA calculation: {e}"
                )
                raise DataStructureError(error_msg)

        else:
            error_msg = f"For patient's {image_id} image, patient is excluded, only supported PET Units are BQML for Philips, Siemens and GE or CNTS for Philips"
            raise DataStructureError(error_msg)


def apply_suv_correction(dicom_files, suv_image):
    reject_unsupported_enhanced_pet(dicom_files)

    def process_single_slice(dicom_file_path):
        ds = pydicom.dcmread(dicom_file_path)

        def get_tracer_name(rph_item):
            value = getattr(rph_item, "Radiopharmaceutical", None)
            if value is None and (0x0018, 0x0031) in rph_item:
                value = rph_item[(0x0018, 0x0031)].value
            return str(value) if value is not None else None

        def get_datetime_on_acquisition_day(time_value, acquisition_time):
            return _datetime_on_reference_date(time_value, acquisition_time)

        def compute_elapsed_time_for_start_decay_correction(
            ds,
            injection_time,
            acquisition_time,
            decay_constant,
        ):
            manufacturer = ds.Manufacturer.upper()

            series_time = get_datetime_on_acquisition_day(
                ds.SeriesTime,
                acquisition_time,
            )

            if "PHILIPS" in manufacturer:
                if acquisition_time == series_time:
                    return (acquisition_time - injection_time).total_seconds()
                return calc_elapsed_time(ds, decay_constant, acquisition_time, injection_time)

            if "SIEMENS" in manufacturer or "CPS" in manufacturer or "CTI" in manufacturer:
                try:
                    private_time = get_datetime_on_acquisition_day(
                        ds[(0x0071, 0x1022)].value,
                        acquisition_time,
                    )
                    return (private_time - injection_time).total_seconds()
                except (KeyError, TypeError):
                    if acquisition_time == series_time:
                        return (acquisition_time - injection_time).total_seconds()
                    return calc_elapsed_time(ds, decay_constant, acquisition_time, injection_time)

            if "GE" in manufacturer:
                try:
                    private_time = get_datetime_on_acquisition_day(
                        ds[(0x0009, 0x100D)].value,
                        acquisition_time,
                    )
                    return (private_time - injection_time).total_seconds()
                except (KeyError, TypeError):
                    if acquisition_time == series_time:
                        return (acquisition_time - injection_time).total_seconds()
                    frame_reference_time = float(ds.FrameReferenceTime) / 1000.0
                    return (acquisition_time - injection_time).total_seconds() - frame_reference_time

            if acquisition_time == series_time:
                return (acquisition_time - injection_time).total_seconds()
            return calc_elapsed_time(ds, decay_constant, acquisition_time, injection_time)

        def process_gml(pixel_array_units, ds):
            suv_type, factor = get_gml_normalization_info(ds)
            patient_weight = float(ds.PatientWeight)

            if suv_type == "BW":
                return pixel_array_units

            return pixel_array_units * (patient_weight / factor)

        def process_cm2ml(pixel_array_units, ds):
            suv_type = ds.get((0x0054, 0x1006), None)
            if suv_type is not None and suv_type.value != "BSA":
                raise DataStructureError(f"CM2ML with {suv_type.value} SUV normalization is not supported!")

            patient_weight = float(ds.PatientWeight)
            height_cm = get_patient_height_cm(ds)
            bsa_m2 = calculate_bsa_du_bois(height_cm, patient_weight)

            return pixel_array_units * (patient_weight / (bsa_m2 * 10.0))

        def process_bqml(activity_concentration, ds):
            rph = ds.RadiopharmaceuticalInformationSequence[0]
            patient_weight = float(ds.PatientWeight)
            injected_dose = float(rph.RadionuclideTotalDose)

            tracer_name = get_tracer_name(rph)
            if tracer_name is not None and is_fdg(tracer_name) and injected_dose < 10000:
                injected_dose *= 1000000
                warnings.warn(
                    f"Injected dose is {injected_dose} Bq, it is too low for FDG, assumed to be in MBq",
                    DataStructureWarning,
                )

            if injected_dose <= 0:
                raise DataStructureError("The injected PET tracer dose is zero.")

            half_life = get_radionuclide_half_life(ds)
            decay_constant = np.log(2) / half_life
            decay_correction = ds.DecayCorrection

            if decay_correction == "ADMIN":
                return activity_concentration / (injected_dose / (patient_weight * 1000))

            injection_time, acquisition_time = resolve_injection_and_acquisition_times(
                ds,
                half_life,
                decay_constant,
            )

            if decay_correction == "START":
                elapsed_time = compute_elapsed_time_for_start_decay_correction(
                    ds,
                    injection_time,
                    acquisition_time,
                    decay_constant,
                )
                decay_factor = np.exp(-(np.log(2) * elapsed_time) / half_life)
                decay_corrected_dose = injected_dose * decay_factor
                return activity_concentration / (decay_corrected_dose / (patient_weight * 1000))

            if decay_correction == "NONE":
                decay_during_frame = decay_constant * float(ds.ActualFrameDuration) / 1000.0
                avg_count_rate_time = (1 / decay_constant) * np.log(
                    decay_during_frame / (1 - np.exp(-decay_during_frame))
                )
                decay_corrected_activity_concentration = activity_concentration * np.exp(
                    decay_constant * ((acquisition_time - injection_time).total_seconds() + avg_count_rate_time)
                )

                return decay_corrected_activity_concentration / (injected_dose / (patient_weight * 1000))

            raise DataStructureError(f"Decay correction {decay_correction} is not supported!")

        def process_cnts(pixel_array_units, ds):
            manufacturer = ds.Manufacturer.upper()
            if "PHILIPS" not in manufacturer:
                raise DataStructureError(f"Vendor {ds.Manufacturer} is not supported with CNTS units!")

            if (0x7053, 0x1009) in ds and ds[(0x7053, 0x1009)].value != 0:
                activity_concentration_bqml = pixel_array_units * ds[(0x7053, 0x1009)].value
                return process_bqml(activity_concentration_bqml, ds)

            if ds.DecayCorrection != "NONE" and (0x7053, 0x1000) in ds and ds[(0x7053, 0x1000)].value != 0:
                return pixel_array_units * ds[(0x7053, 0x1000)].value

            raise DataStructureError("Philips-specific scaling factors not present!")

        units = ds.Units
        pixel_array_units = (ds.pixel_array * ds.RescaleSlope) + ds.RescaleIntercept

        if units == "GML":
            suv = process_gml(pixel_array_units, ds)
        elif units == "CM2ML":
            suv = process_cm2ml(pixel_array_units, ds)
        elif units == "BQML":
            suv = process_bqml(pixel_array_units, ds)
        elif units == "CNTS":
            suv = process_cnts(pixel_array_units, ds)
        else:
            raise DataStructureError(f"Units {units} are not supported!")

        return suv.T

    intensity_array = np.zeros(suv_image.GetSize())

    dicom_files = sorted(dicom_files, key=lambda f: float(f["ds"].ImagePositionPatient[2]))
    for z_slice_id, dicom_file in enumerate(dicom_files):
        intensity_array[:, :, z_slice_id] = process_single_slice(dicom_file["file_path"])

    intensity_image = sitk.GetImageFromArray(intensity_array.T)
    intensity_image.SetOrigin(suv_image.GetOrigin())
    intensity_image.SetSpacing(np.array(suv_image.GetSpacing()))
    intensity_image.SetDirection(np.array(suv_image.GetDirection()))
    return intensity_image
