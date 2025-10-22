import streamlit as st
import pandas as pd
from pandas.api.types import CategoricalDtype
from datetime import time, datetime
import calendar

# Workaround to deal with relative import issues
# https://discuss.streamlit.io/t/importing-modules-in-pages/26853/2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _state_control import setup_state, reset_to_defaults, DEFAULT_INPUTS

st.set_page_config(layout="wide")

with open("app/style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

setup_state()

from streamlit_extras.stylable_container import stylable_container
from utils import Utils
from _app_utils import (
    get_text,
    get_text_sheet,
    COLORSCHEME,
    MONTH_MAPPING,
    REVERSE_MONTH_MAPPING,
    get_rota_month_strings,
)

u = Utils()

text_df = get_text_sheet("setup")

st.session_state["visited_setup_page"] = True

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title(get_text("page_title", text_df))

with col2:
    st.image("app/assets/daa-logo.svg", width=300)

st.caption(get_text("page_description", text_df))

st.button(
    get_text("reset_parameters_button", text_df),
    type="primary",
    on_click=reset_to_defaults,
    icon=":material/history:",
)
st.caption(get_text("reset_parameters_warning", text_df))

st.divider()

st.header("HEMS Rota Builder")


@st.fragment
def rota_start_end_dates():
    st.markdown("### Summer and Winter Setup")

    col_summer_start, col_summer_end, col_summer_spacing = st.columns(3)

    with col_summer_start:
        start_month = st.selectbox(
            "Select **start** month (inclusive) for Summer Rota",
            list(MONTH_MAPPING.keys()),
            index=st.session_state.summer_start_month_index,
            key="key_summer_start_month_index",
            on_change=lambda: setattr(
                st.session_state,
                "summer_start_month_index",
                # note index is 1 less than actual month due to zero indexing in python
                MONTH_MAPPING[st.session_state.key_summer_start_month_index] - 1,
            ),
        )
    with col_summer_end:
        end_month = st.selectbox(
            "Select **end** month (inclusive) for Summer Rota",
            list(MONTH_MAPPING.keys()),
            index=st.session_state.summer_end_month_index,
            key="key_summer_end_month_index",
            on_change=lambda: setattr(
                st.session_state,
                "summer_end_month_index",
                # note index is 1 less than actual month due to zero indexing in python
                MONTH_MAPPING[st.session_state.key_summer_end_month_index] - 1,
            ),
        )

        (
            start_month_num,
            end_month_num,
            summer_start_date,
            summer_end_date,
            summer_end_day,
            winter_start_date,
            winter_end_date,
            winter_end_day,
        ) = get_rota_month_strings(start_month, end_month)

    if start_month_num <= end_month_num:
        # Output
        st.write(
            f"☀️ Summer rota runs from {summer_start_date} to {summer_end_date} (inclusive)"
        )
        st.write(
            f"❄️ Winter rota runs from {winter_start_date} to {winter_end_date} (inclusive)"
        )

        pd.DataFrame(
            [
                {"what": "summer_start_month", "month": start_month_num},
                {"what": "summer_end_month", "month": end_month_num},
                {"what": "summer_start_month_string", "month": start_month},
                {"what": "summer_end_month_string", "month": end_month},
            ]
        ).to_csv("actual_data/rota_start_end_months.csv", index=False)
    else:
        default_start_month = DEFAULT_INPUTS["summer_start_month_index"] + 1
        default_end_month = DEFAULT_INPUTS["summer_end_month_index"] + 1
        default_start_month_name = REVERSE_MONTH_MAPPING[default_start_month]
        default_end_month_name = REVERSE_MONTH_MAPPING[default_end_month]

        default_summer_end_day = calendar.monthrange(2024, end_month_num)[
            1
        ]  # Assume leap year for Feb
        default_summer_end_date = f"{default_summer_end_day}th {default_end_month_name}"
        st.error(
            f"""End month must be later than start month. Using default summer start of 1st {default_start_month_name} and summer end of {default_summer_end_date}."""
        )
        pd.DataFrame(
            [
                {"what": "summer_start_month", "month": default_start_month},
                {"what": "summer_end_month", "month": default_end_month},
            ]
        ).to_csv("actual_data/rota_start_end_months.csv", index=False)


rota_start_end_dates()


@st.fragment
def fleet_setup():
    st.markdown(f"""### {get_text("header_fleet_setup", text_df)}""")

    st.caption("""At present, the fleet cannot be expanded beyond what is currently
               available, though resources can be removed. """)

    col_1_fleet_setup, col_2_fleet_setup, blank_col_fleet_setup = st.columns(3)

    with col_1_fleet_setup:
        num_helicopters = st.number_input(
            get_text("set_num_helicopters", text_df),
            min_value=1,
            max_value=2,
            disabled=False,
            value=st.session_state.num_helicopters,
            help=get_text("help_helicopters", text_df),
            on_change=lambda: setattr(
                st.session_state,
                "num_helicopters",
                st.session_state.key_num_helicopters,
            ),
            key="key_num_helicopters",
        )

    with col_2_fleet_setup:
        num_cars = st.number_input(
            get_text("set_num_additional_cars", text_df),
            min_value=0,
            max_value=1,
            disabled=False,
            value=st.session_state.num_cars,
            help=get_text("help_cars", text_df),
            on_change=lambda: setattr(
                st.session_state, "num_cars", st.session_state.key_num_cars
            ),
            key="key_num_cars",
        )

    # Pull in the callsign and model lookup
    # For each, we will pull through the edited dataframe and the default dataframe
    # This allows us to ensure we are able to add in information about any default resources
    # even if we have removed it from the saved non-default lookups
    callsign_lookup = pd.read_csv("actual_data/callsign_registration_lookup.csv")
    callsign_lookup_default = pd.read_csv(
        "actual_data/callsign_registration_lookup_DEFAULT.csv"
    )
    callsign_lookup_columns = callsign_lookup.columns

    models = pd.read_csv("actual_data/service_schedules_by_model.csv")
    models_default = pd.read_csv("actual_data/service_schedules_by_model_DEFAULT.csv")
    models_columns = models.columns

    hems_rota_default = pd.read_csv("actual_data/HEMS_ROTA_DEFAULT.csv")

    callsign_lookup = pd.concat([callsign_lookup, callsign_lookup_default]).reset_index(
        drop=True
    )
    callsign_lookup = callsign_lookup.drop_duplicates(keep="first")
    print(callsign_lookup)

    models = pd.concat([models, models_default]).reset_index(drop=True)
    models = models.drop_duplicates(keep="first", subset=["model", "vehicle_type"])
    model_options = list(
        models[models["vehicle_type"] == "helicopter"]["model"].unique()
    )
    print(models)

    potential_fleet = callsign_lookup.merge(models, how="left", on=["model"])

    potential_fleet["callsign_group"] = potential_fleet["callsign"].str.extract(
        r"(\d+)"
    )

    potential_fleet["callsign_count"] = potential_fleet.groupby("callsign_group")[
        "callsign_group"
    ].transform("count")

    default_helos = potential_fleet[potential_fleet["vehicle_type"] == "helicopter"]

    default_cars = potential_fleet[
        (potential_fleet["vehicle_type"] == "car")
        & (potential_fleet["callsign_count"] == 1)
    ]

    default_helos = default_helos.head(num_helicopters)
    default_helos["has_car"] = True

    default_cars = default_cars.head(num_cars)

    st.markdown("#### Define the Helicopters")
    st.caption(
        "Columns with the :material/edit_note: symbol can be edited by double clicking the relevant table cell."
    )

    updated_helos_df = st.data_editor(
        default_helos,
        hide_index=True,
        key="helicopter_data_editor",
        column_order=[
            "callsign",
            #   "callsign_group",
            "registration",
            "has_car",
            "model",  # "service_schedule_months", "service_duration_weeks"
        ],
        column_config={
            "registration": st.column_config.TextColumn(
                label="Registration", required=True
            ),
            "callsign": st.column_config.TextColumn(label="Callsign", required=True),
            "callsign_group": st.column_config.TextColumn(
                label="Callsign", disabled=True
            ),
            "has_car": st.column_config.CheckboxColumn(label="Has a Backup Car"),
            "model": st.column_config.SelectboxColumn(
                label="Model", options=model_options, required=True
            ),
            # "service_schedule_months": st.column_config.NumberColumn(label="Servicing Interval (Months)", disabled=True),
            # "service_duration_weeks": st.column_config.NumberColumn(label="Servicing Interval (Weeks)", disabled=True)
        },
    )

    st.markdown("#### Define the Backup Cars")

    backup_cars = potential_fleet[
        (potential_fleet["vehicle_type"] == "car")
        & (potential_fleet["callsign_count"] > 1)
        & (potential_fleet["callsign_group"]).isin(
            updated_helos_df[updated_helos_df["has_car"] == True][
                "callsign_group"
            ].unique()
        )
    ]

    updated_backup_cars_df = st.data_editor(
        backup_cars,
        hide_index=True,
        key="backup_car_data_editor",
        column_order=[
            "callsign",
            #   "callsign_group",
            "registration",
            "model",  # "service_schedule_months", "service_duration_weeks"
        ],
        column_config={
            "registration": st.column_config.TextColumn(
                label="Registration", required=True
            ),
            "callsign": st.column_config.TextColumn(label="Callsign", required=True),
            "callsign_group": st.column_config.TextColumn(
                label="Callsign", disabled=True
            ),
            "model": st.column_config.SelectboxColumn(
                label="Model",
                options=potential_fleet[potential_fleet["vehicle_type"] == "car"][
                    "model"
                ].unique(),
                required=True,
            ),
            # "service_schedule_months": st.column_config.NumberColumn(label="Servicing Interval (Months)", disabled=True),
            # "service_duration_weeks": st.column_config.NumberColumn(label="Servicing Interval (Weeks)", disabled=True)
        },
    )

    st.markdown("#### Define the Standalone Cars")
    st.caption(
        "Columns with the :material/edit_note: symbol can be edited by double clicking the relevant table cell."
    )

    updated_cars_df = st.data_editor(
        default_cars,
        hide_index=True,
        key="standalone_car_data_editor",
        column_order=[
            "callsign",
            #   "callsign_group",
            "registration",
            "model",
            #   "service_schedule_months", "service_duration_weeks"
        ],
        column_config={
            "registration": st.column_config.TextColumn(
                label="Registration", required=True
            ),
            "callsign": st.column_config.TextColumn(label="Callsign", required=True),
            # "callsign_group": st.column_config.TextColumn(label="Callsign", disabled=True),
            "model": st.column_config.SelectboxColumn(
                label="Model",
                options=potential_fleet[potential_fleet["vehicle_type"] == "car"][
                    "model"
                ].unique(),
                required=True,
            ),
            # "service_schedule_months": st.column_config.NumberColumn(label="Servicing Interval (Months)", disabled=True),
            # "service_duration_weeks": st.column_config.NumberColumn(label="Servicing Interval (Weeks)", disabled=True)
        },
    )

    final_df = pd.concat([updated_helos_df, updated_backup_cars_df, updated_cars_df])

    final_df[callsign_lookup_columns].drop_duplicates().to_csv(
        "actual_data/callsign_registration_lookup.csv", index=False
    )

    # final_df[models_columns].drop_duplicates().to_csv("actual_data/service_schedules_by_model.csv", index=False)

    # hems_rota_default['callsign_group'] = hems_rota_default['callsign_group'].astype('str')
    hems_rota = hems_rota_default[
        hems_rota_default["callsign"].isin(final_df.callsign.unique())
    ]
    hems_rota.to_csv("actual_data/HEMS_ROTA.csv", index=False)

    with st.expander("Click here to view the final fleet dataframes"):
        st.markdown("### Callsign Lookup")
        st.dataframe(
            final_df[callsign_lookup_columns].drop_duplicates(), hide_index=True
        )
        st.markdown("### Vehicle Model Details")
        st.dataframe(
            pd.read_csv("actual_data/service_schedules_by_model.csv"), hide_index=True
        )

    st.markdown("### Individual Rota Setup")

    # Load callsign registration (this should reflect the output of fleet_setup)
    try:
        callsign_registration_lookup_df = pd.read_csv(
            "actual_data/callsign_registration_lookup.csv"
        )
    except FileNotFoundError:
        st.error(
            "actual_data/callsign_registration_lookup.csv not found. Please run Fleet Setup first."
        )
        st.stop()

    # Derive callsign_group and vehicle_type if not already perfect from the CSV
    # (The fleet_setup already does this, so it might be redundant if CSV is always up-to-date)
    if "callsign_group" not in callsign_registration_lookup_df.columns:
        callsign_registration_lookup_df["callsign_group"] = (
            callsign_registration_lookup_df["callsign"].str.extract(r"(\d+)")
        )
    if (
        "vehicle_type" not in callsign_registration_lookup_df.columns
        and "model" in callsign_registration_lookup_df.columns
    ):
        # This logic might need to align with how vehicle_type is determined in fleet_setup
        # For simplicity, let's assume it's present or use a simplified model-based inference
        callsign_registration_lookup_df["vehicle_type"] = (
            callsign_registration_lookup_df["model"].apply(
                lambda x: "helicopter"
                if isinstance(x, str) and "Airbus" in x or "H1" in x
                else "car"  # Simplified
            )
        )

    # Load default rota (this is the master template for shifts)
    try:
        df_default_rota = pd.read_csv("actual_data/HEMS_ROTA_DEFAULT.csv")
    except FileNotFoundError:
        st.error("HEMS_ROTA_DEFAULT.csv not found!")
        st.stop()

    # Sort: group, then helicopter first
    # Ensure 'vehicle_type' exists before trying to use it for sorting
    if "vehicle_type" in callsign_registration_lookup_df.columns:
        vehicle_order = CategoricalDtype(categories=["helicopter", "car"], ordered=True)
        callsign_registration_lookup_df["vehicle_type"] = (
            callsign_registration_lookup_df["vehicle_type"].astype(vehicle_order)
        )
        sorted_lookup_df = callsign_registration_lookup_df.sort_values(
            by=["callsign_group", "vehicle_type"]
        )
    else:
        st.warning(
            "'vehicle_type' column missing in callsign lookup. Rota setup might be incomplete."
        )
        sorted_lookup_df = (
            callsign_registration_lookup_df.copy()
        )  # Proceed without vehicle type sorting if missing

    rota_data = {}
    helicopter_rotas_by_group = {}

    # Define columns for the rota editor UI and their configuration
    editable_rota_columns = [
        "category",
        "summer_start",
        "summer_end",
        "winter_start",
        "winter_end",
    ]
    rota_editor_column_config = {
        "category": st.column_config.SelectboxColumn(
            label="Category", options=["CC", "EC"], required=True
        ),
        "summer_start": st.column_config.NumberColumn(
            label="Summer Start Hour", min_value=0, max_value=23, step=1, required=True
        ),
        "summer_end": st.column_config.NumberColumn(
            label="Summer End Hour", min_value=0, max_value=23, step=1, required=True
        ),
        "winter_start": st.column_config.NumberColumn(
            label="Winter Start Hour", min_value=0, max_value=23, step=1, required=True
        ),
        "winter_end": st.column_config.NumberColumn(
            label="Winter End Hour", min_value=0, max_value=23, step=1, required=True
        ),
    }
    # Column order for the editor will be just the editable_rota_columns
    rota_editor_column_order = editable_rota_columns[:]

    for idx, row_lookup in sorted_lookup_df.iterrows():
        callsign = row_lookup["callsign"]
        model = row_lookup.get(
            "model", "N/A"
        )  # Use .get for safety if 'model' might be missing
        # Ensure vehicle_type and callsign_group are present in row_lookup
        vehicle_type = row_lookup.get("vehicle_type", "unknown")
        group = row_lookup.get("callsign_group", "unknown_group")

        if vehicle_type == "unknown" or group == "unknown_group":
            st.warning(
                f"Skipping {callsign} due to missing vehicle_type or callsign_group in lookup data."
            )
            continue

        st.markdown(f"#### Set up rota for {callsign} ({model})")

        # Prepare existing_rota_for_resource: contains all columns (identifiers + schedule)
        existing_rota_for_resource = df_default_rota[
            df_default_rota["callsign"] == callsign
        ].copy()

        if existing_rota_for_resource.empty:
            num_rows_input = st.number_input(
                f"Number of shifts for {callsign}",
                min_value=1,
                max_value=5,
                value=2,
                key=f"{callsign}_num_rows",
            )
            default_category_val = "EC" if vehicle_type == "helicopter" else "CC"
            existing_rota_for_resource = pd.DataFrame(
                {
                    "callsign": [callsign] * num_rows_input,
                    "category": [default_category_val] * num_rows_input,
                    "vehicle_type": [vehicle_type] * num_rows_input,
                    "callsign_group": [group] * num_rows_input,
                    "summer_start": [7] * num_rows_input,
                    "winter_start": [7] * num_rows_input,
                    "summer_end": [19] * num_rows_input,
                    "winter_end": [19] * num_rows_input,
                }
            )

        # This DataFrame will be passed to st.data_editor, containing only schedule columns
        data_for_rota_editor = existing_rota_for_resource[editable_rota_columns].copy()

        current_edited_df = None  # This will hold the final DataFrame for this resource (with all columns)

        if vehicle_type == "car" and group in helicopter_rotas_by_group:
            toggle_key = f"{callsign}_same_as_heli"
            # Initialize session state for the toggle if not already present
            if toggle_key not in st.session_state:
                st.session_state[toggle_key] = True  # Default to using heli rota

            use_heli_rota = st.toggle(
                f"Use same rota as helicopter for group {group}?", key=toggle_key
            )

            if use_heli_rota:
                # helicopter_rotas_by_group stores the full DataFrame (identifiers + schedule)
                full_heli_rota_template = helicopter_rotas_by_group[group].copy()

                # Update identifiers for the car
                full_heli_rota_template["callsign"] = callsign
                full_heli_rota_template["vehicle_type"] = "car"
                full_heli_rota_template["callsign_group"] = (
                    group  # Should be the same group
                )

                current_edited_df = full_heli_rota_template
                st.info(f"Using helicopter rota for {callsign} ({model}).")
            else:
                # Custom rota for the car - show editor with only schedule columns
                edited_schedule_df = st.data_editor(
                    data_for_rota_editor,  # Contains only editable_rota_columns
                    column_order=rota_editor_column_order,
                    column_config=rota_editor_column_config,
                    hide_index=True,
                    num_rows="dynamic",
                    key=f"{callsign}_custom_car_rota_editor",
                )
                # Reconstruct the full DataFrame
                current_edited_df = edited_schedule_df.copy()
                current_edited_df["callsign"] = callsign
                current_edited_df["vehicle_type"] = "car"  # Explicitly car
                current_edited_df["callsign_group"] = group
        else:
            # Default editor for helicopters, or cars not having the sync toggle
            edited_schedule_df = st.data_editor(
                data_for_rota_editor,  # Contains only editable_rota_columns
                column_order=rota_editor_column_order,
                column_config=rota_editor_column_config,
                hide_index=True,
                num_rows="dynamic",
                key=f"{callsign}_default_rota_editor",
            )
            # Reconstruct the full DataFrame
            current_edited_df = edited_schedule_df.copy()
            current_edited_df["callsign"] = callsign
            current_edited_df["vehicle_type"] = (
                vehicle_type  # This vehicle's actual type
            )
            current_edited_df["callsign_group"] = group

        rota_data[callsign] = current_edited_df

        if vehicle_type == "helicopter":
            # current_edited_df is the helicopter's fully reconstructed rota
            helicopter_rotas_by_group[group] = current_edited_df.copy()

    st.markdown("## Full Rota Preview")
    if rota_data:
        full_rota_df = pd.concat(rota_data.values(), ignore_index=True)

        # Ensure all core columns from df_default_rota are present and in the correct order
        # The columns 'callsign', 'vehicle_type', 'callsign_group' should now be correctly
        # populated from the loop logic.

        # Make sure all columns from the default rota schema are present
        for col in df_default_rota.columns:
            if col not in full_rota_df.columns:
                # Add missing columns with appropriate default (e.g., NA or specific values)
                # This handles cases where df_default_rota might have more columns than generated.
                full_rota_df[col] = pd.NA
                if (
                    col == "category" and "category" not in full_rota_df.columns
                ):  # Example default
                    full_rota_df[col] = "EC"  # Or some other logic for default category

        # Select and order columns according to the df_default_rota schema
        # Filter full_rota_df.columns to only those that are also in df_default_rota.columns
        # to prevent errors if full_rota_df accidentally gains extra columns not in the schema.
        final_columns_ordered = [
            col for col in df_default_rota.columns if col in full_rota_df.columns
        ]
        full_rota_df = full_rota_df[final_columns_ordered]

        st.dataframe(full_rota_df, hide_index=True)
        # Save to HEMS_ROTA.csv (the working file), not HEMS_ROTA_DEFAULT.csv
        full_rota_df.to_csv("actual_data/HEMS_ROTA.csv", index=False)
        st.success("Final rota automatically saved to HEMS_ROTA.csv!")
    else:
        st.warning("No rota data generated to preview or save.")


fleet_setup()

# col_summer, col_winter, col_summer_winter_spacing = st.columns(3)
# with col_summer:
#     st.caption(get_text("summer_rota_help", text_df))
# with col_winter:
#     st.caption(get_text("winter_rota_help", text_df))


# @st.fragment
# def fleet_editors(final_helo_df, final_car_df):
#     # Create an editable dataframe for people to modify the parameters in
#     st.markdown("##### Helicopters")

#     st.info("Single resources with different levels of care at different times will be represented as two rows with non-overlapping rota times")

#     st.caption("Columns with the :material/edit_note: symbol can be edited by double clicking the relevant table cell.")

#     updated_helo_df = st.data_editor(
#         final_helo_df.reset_index(),
#         disabled=["vehicle_type"],
#         hide_index=True,
#         column_order=["vehicle_type", "callsign", "registration", "category", "model",
#                     "summer_start", "summer_end", "winter_start", "winter_end"],
#         column_config={
#             "vehicle_type": "Vehicle Type",
#             "callsign": st.column_config.TextColumn(
#                 "Callsign", disabled=True
#                 ),
#             "registration": st.column_config.TextColumn(
#                 "Registration", disabled=True
#             ),
#             "category": st.column_config.SelectboxColumn(
#                 "Care Type",
#                 options=["EC", "CC"],
#                 disabled=False
#                 ),
#             "model": st.column_config.SelectboxColumn(
#                 "Model",
#                 options=["Airbus EC135", "Airbus H145"],
#             ),
#             "summer_start": st.column_config.TimeColumn(
#                 "Summer Start", format="HH:mm",
#                 disabled=False
#             ),
#             "summer_end": st.column_config.TimeColumn(
#                 "Summer End", format="HH:mm",
#                 disabled=False
#             ),
#             "winter_start": st.column_config.TimeColumn(
#                 "Winter Start", format="HH:mm",
#                 disabled=False
#             ),
#             "winter_end": st.column_config.TimeColumn(
#                 "Winter End", format="HH:mm",
#                 disabled=False
#             )
#             }
#         )

#     st.caption("""
# :red_car: **All helicopters in the model are automatically assumed to have a backup car assigned to them for use
# when the helicopter is unavailable for any reason.**
# """)


#     st.markdown("##### Additional Cars")
#     st.caption("""
# In the table below you can also alter the parameters of the *additional* cars that have their own separate callsign
# group and operate as a totally separate resource to the helicopters.""")
#     st.caption("Columns with the :material/edit_note: symbol can be edited by double clicking the relevant table cell.")


#     final_car_df["vehicle_type"] = final_car_df["vehicle_type"].apply(lambda x: x.title())

#     updated_car_df = st.data_editor(final_car_df.reset_index(),
#                                     hide_index=True,
#                                     disabled=["vehicle_type"],
#                                     column_order=["vehicle_type", "callsign", "registration", "category",
#                                                   "model", "summer_start", "summer_end",
#                                                   "winter_start", "winter_end"],
#                                     column_config={
#             "vehicle_type": "Vehicle Type",
#             "callsign": st.column_config.TextColumn(
#                 "Callsign", disabled=True
#                 ),
#             "registration": st.column_config.TextColumn(
#                 "Registration", disabled=True
#             ),
#             "category": st.column_config.SelectboxColumn(
#                 "Care Type", options=["EC", "CC"],
#                 disabled=False
#             ),
#                 "model": st.column_config.SelectboxColumn(
#                 "Model", options=["Volvo XC90"],
#                 disabled=True
#             ),
#             "summer_start": st.column_config.TimeColumn(
#                 "Summer Start", format="HH:mm",
#                 disabled=False
#             ),
#             "summer_end": st.column_config.TimeColumn(
#                 "Summer End", format="HH:mm",
#                 disabled=False
#             ),
#             "winter_start": st.column_config.TimeColumn(
#                 "Winter Start", format="HH:mm",
#                 disabled=False
#             ),
#             "winter_end": st.column_config.TimeColumn(
#                 "Winter End", format="HH:mm",
#                 disabled=False
#             )

#             }
#                                     )

#     # Join the dataframes back together
#     final_rota = pd.concat([updated_helo_df, updated_car_df]).drop(columns='index')

#     # Convert vehicle type column back to expected capitalisation
#     final_rota["vehicle_type"] = final_rota["vehicle_type"].str.lower()

#     # Add callsign group column back in
#     final_rota["callsign_group"] = final_rota["callsign"].str.extract("(\d+)")

#     # print(final_rota)

#     # # Merge with service schedule df to get actual servicing intervals for chosen model
#     # final_rota = final_rota.merge(
#     #     u.SERVICING_SCHEDULES_BY_MODEL.merge(
#     #         pd.read_csv("actual_data/callsign_registration_lookup.csv"), how="left", on="model"),
#     #     on=["model","registration"], how="left"
#     #     )

#     print("Final Rota - Before Companion Cars")
#     print(final_rota)

#     ###############
#     # Companion Cars
#     ###############

#     # Take a copy of the helicopter df to allow us to create the cars that go alongside it
#     # We can assume operating hours and care category will be the same
#     companion_car_df = updated_helo_df.copy()
#     print("Initial Companion Car df")
#     print(companion_car_df)

#     # TODO: For now, we have hardcoded companion cars to be Volvo XC90s
#     companion_car_df["model"] = "Volvo XC90"
#     # Register them as cars instead of helicopters
#     companion_car_df["vehicle_type"] = "car"
#     # Update callsign
#     companion_car_df["callsign"] = companion_car_df["callsign"].str.replace("H", "CC")
#     # Add callsign group column
#     companion_car_df["callsign_group"] = companion_car_df["callsign"].str.extract("(\d+)")
#     # Remove 'last_service' date
#     # SR UPDATE 26/3 - no longer needed due to RP redesign of this df
#     # companion_car_df = companion_car_df.drop(columns=["last_service"])

#     # Merge with service schedule df to get actual servicing intervals for chosen model
#     companion_car_df.drop(columns=["service_schedule_months", "service_duration_weeks"])

#     companion_car_df = companion_car_df.merge(
#         u.SERVICING_SCHEDULES_BY_MODEL,
#         on=["model","vehicle_type"], how="left"
#         )

#     # Join this onto the list of helicopters and separate cars, then sort
#     final_rota = (pd.concat([final_rota, companion_car_df])
#                   .sort_values(["callsign_group", "vehicle_type"], ascending=[True, False])
#                   .drop(columns='index')
#                   )

#     # Remove the servicing columns as they will reflect the originally set models in the
#     # default rota
#     # SR Comment 26/3 - testing removal after RP redesign of service schedule monitoring
#     # final_rota = final_rota.drop(columns=["service_schedule_months","service_duration_weeks"])

#     print("Generated Rota")
#     print(final_rota)

#     print("Servicing Schedules by Model")
#     print(u.SERVICING_SCHEDULES_BY_MODEL)
#     print("Final Rota after Merge with Servicing Schedules")
#     print(final_rota)

#     # Convert the time columns back to something the model can understand
#     for col in ["summer_start", "winter_start", "summer_end", "winter_end"]:
#         final_rota[col] = final_rota[col].apply(lambda x: x.hour)

#     # Sort the columns into the order of the original rota
#     # Write back
#     final_rota_cols = pd.read_csv('actual_data/HEMS_ROTA_DEFAULT.csv').columns
#     # Write the rota back to a csv
#     final_rota[final_rota_cols].to_csv('actual_data/HEMS_ROTA.csv', index=False)

#     callsign_registration_lookup_cols = pd.read_csv('actual_data/callsign_registration_lookup_DEFAULT.csv').columns
#     callsign_output = final_rota[callsign_registration_lookup_cols].drop_duplicates()
#     callsign_output['registration'] = callsign_output.apply(lambda x: x["callsign"].lower() if "CC" in x["callsign"] else x["registration"], axis=1)
#     callsign_output.to_csv('actual_data/callsign_registration_lookup.csv', index=False)

# fleet_editors(final_helo_df, final_car_df)

# st.divider()

# st.header("Demand Parameters")

# st.caption("""
# At present it is only possible to apply an overall demand adjustment, which increases the number
# of calls that will be received per day in the model. You can use the slider below to carry out
# this adjustment.

# In future, the model will allow more granular control of additional demand.
# """)

# demand_adjust_type = "Overall Demand Adjustment"

# # TODO: Add to session state
# # demand_adjust_type = st.radio("Adjust High-level Demand",
# #          ["Overall Demand Adjustment",
# #           "Per Season Demand Adjustment",
# #           "Per AMPDS Code Demand Adjustment"],
# #           key="demand_adjust_type",
# #           horizontal=True,
# #           disabled=True
# #           )

# if demand_adjust_type == "Overall Demand Adjustment":
#     overall_demand_mult = st.slider(
#         "Overall Demand Adjustment",
#         min_value=90,
#         max_value=200,
#         value=st.session_state.overall_demand_mult,
#         format="%d%%",
#         on_change= lambda: setattr(st.session_state, 'overall_demand_mult', st.session_state.key_overall_demand_mult),
#         key="key_overall_demand_mult"
#         )

overall_demand_mult = 100

# elif demand_adjust_type == "Per Season Demand Adjustment":
#     season_demand_col_1, season_demand_col_2, season_demand_col_3, season_demand_col_4 = st.columns(4)

#     spring_demand_mult = season_demand_col_1.slider(
#         "🌼 Spring Demand Adjustment",
#         min_value=90,
#         max_value=200,
#         value=st.session_state.spring_demand_mult,
#         format="%d%%",
#         on_change= lambda: setattr(st.session_state, 'spring_demand_mult', st.session_state.key_spring_demand_mult),
#         key="key_spring_demand_mult"
#         )

#     summer_demand_mult = season_demand_col_2.slider(
#         "☀️ Summer Demand Adjustment",
#         min_value=90,
#         max_value=200,
#         value=st.session_state.summer_demand_mult,
#         format="%d%%",
#         on_change= lambda: setattr(st.session_state, 'summer_demand_mult', st.session_state.key_summer_demand_mult),
#         key="key_summer_demand_mult"
#         )

#     autumn_demand_mult = season_demand_col_3.slider(
#         "🍂 Autumn Demand Adjustment",
#         min_value=90,
#         max_value=200,
#         value=st.session_state.autumn_demand_mult,
#         format="%d%%",
#         on_change= lambda: setattr(st.session_state, 'autumn_demand_mult', st.session_state.key_autumn_demand_mult),
#         key="key_autumn_demand_mult"
#         )

#     winter_demand_mult = season_demand_col_4.slider(
#         "❄️ Winter Demand Adjustment",
#         min_value=90,
#         max_value=200,
#         value=st.session_state.winter_demand_mult,
#         format="%d%%",
#         on_change= lambda: setattr(st.session_state, 'winter_demand_mult', st.session_state.key_winter_demand_mult),
#         key="key_winter_demand_mult"
#         )

# elif demand_adjust_type == "Per AMPDS Code Demand Adjustment":
#     st.write("Coming Soon!")

# else:
#     st.error("TELL A DEVELOPER: Check Conditional Code for demand modifier in setup.py")

st.divider()

# st.divider()

st.header(get_text("additional_params_header", text_df))

st.caption(get_text("additional_params_help", text_df))


@st.fragment
def additional_params_expander():
    st.markdown("### Replications")

    number_of_runs_input = st.slider(
        "Number of Runs",
        min_value=1,
        max_value=100,
        value=st.session_state.number_of_runs_input,
        on_change=lambda: setattr(
            st.session_state,
            "number_of_runs_input",
            st.session_state.key_number_of_runs_input,
        ),
        key="key_number_of_runs_input",
        help="""
This controls how many times the simulation will repeat. On each repeat, while the core parameters
will stay the same, slight randomness will occur in the patterns of arrivals and in choices like which
resource responds, what the outcome of each job is, and more.
\n\n
Think of it like watching 10 different versions of the same day in an emergency control room.
In one run, a call comes in five minutes earlier. In another, a different crew is dispatched.
A patient might recover quickly in one case, or need extra help in another.
\n\n
By running the simulation multiple times, we get a better sense of the range of things that *could*
happen — not just a single outcome, but the full picture of what’s likely and what’s possible.
For example, by running 10 replications of 1 year, we can get a better sense of how well the model
will cope in busier and quieter times, helping to understand the likely range of performance that
might be observed in the real world.
""",
    )

    st.markdown("### Simulation and Warm-up Duration")

    col_button_1, col_button_2, col_button_3, col_button_4 = st.columns(4)

    col_button_1.button(
        "Set sim duration to 4 weeks",
        on_click=lambda: setattr(st.session_state, "sim_duration_input", 7 * 4),
    )

    col_button_2.button(
        "Set sim duration to 1 year",
        on_click=lambda: setattr(st.session_state, "sim_duration_input", 365),
    )

    col_button_3.button(
        "Set sim duration to 2 years",
        on_click=lambda: setattr(st.session_state, "sim_duration_input", 365 * 2),
    )

    col_button_4.button(
        "Set sim duration to 3 years",
        on_click=lambda: setattr(st.session_state, "sim_duration_input", 365 * 3),
    )

    sim_duration_input = st.slider(
        "Simulation Duration (days)",
        min_value=1,
        max_value=365 * 3,
        value=st.session_state.sim_duration_input,
        on_change=lambda: setattr(
            st.session_state,
            "sim_duration_input",
            st.session_state.key_sim_duration_input,
        ),
        key="key_sim_duration_input",
    )

    warm_up_duration = st.slider(
        "Warm-up Duration (hours)",
        min_value=0,
        max_value=24 * 10,
        value=st.session_state.warm_up_duration,
        on_change=lambda: setattr(
            st.session_state, "warm_up_duration", st.session_state.key_warm_up_duration
        ),
        key="key_warm_up_duration",
    )

    st.caption(
        f"The simulation will not start recording metrics until {(warm_up_duration / 24):.2f} days have elapsed"
    )

    st.markdown("### Dates")

    st.caption("""
This affects when the simulation will start. The simulation time affects the rota that is used,
as well as the demand (average number of jobs received).
""")

    sim_start_date_input = st.date_input(
        "Select the starting day for the simulation",
        value=st.session_state.sim_start_date_input,
        on_change=lambda: setattr(
            st.session_state,
            "sim_start_date_input",
            st.session_state.key_sim_start_date_input.strftime("%Y-%m-%d"),
        ),
        key="key_sim_start_date_input",
        min_value="2023-01-01",
        max_value="2027-01-01",
    ).strftime("%Y-%m-%d")

    sim_start_time_input = st.time_input(
        "Select the starting time for the simulation",
        value=st.session_state.sim_start_time_input,
        on_change=lambda: setattr(
            st.session_state,
            "sim_start_time_input",
            st.session_state.key_sim_start_time_input.strftime("%H:%M"),
        ),
        key="key_sim_start_time_input",
    ).strftime("%H:%M")

    st.markdown("### Reproducibility and Variation")

    master_seed = st.number_input(
        "Set the master random seed",
        1,
        100000,
        42,
        key="key_master_seed",
        on_change=lambda: setattr(
            st.session_state, "master_seed", st.session_state.key_master_seed
        ),
    )

    st.markdown("### Other Modifiers")

    activity_duration_multiplier = st.slider(
        "Apply a multiplier to activity times",
        value=st.session_state.activity_duration_multiplier,
        max_value=2.0,
        min_value=0.7,
        on_change=lambda: setattr(
            st.session_state,
            "activity_duration_multiplier",
            st.session_state.key_activity_duration_multiplier,
        ),
        key="key_activity_duration_multiplier",
        help="""
This lengthens or shortens all generated activity times by the selected multiplier.
\n\n
For example, if a journey time of 10 minutes was generated, this would be shortened to 8 minutes if
this multiplier is set to 0.8, or lengthened to 12 minutes if the multiplier was set to 1.2.
\n\n
This can be useful for experimenting with the impact of longer or shorter activity times, or for
temporarily adjusting the model if activity times are not accurately reflecting reality.
""",
    )

    create_animation_input = st.toggle(
        "Create Animation",
        value=st.session_state.create_animation_input,
        on_change=lambda: setattr(
            st.session_state,
            "create_animation_input",
            st.session_state.key_create_animation_input,
        ),
        key="key_create_animation_input",
        disabled=True,
        help="Coming soon!",
    )

    amb_data = st.toggle(
        "Model ambulance service data",
        value=st.session_state.amb_data,
        on_change=lambda: setattr(
            st.session_state, "amb_data", st.session_state.key_amb_data
        ),
        key="key_amb_data",
        disabled=True,
        help="Coming soon!",
    )


with st.expander(get_text("additional_params_expander_title", text_df)):
    additional_params_expander()

st.divider()

with st.sidebar:
    with stylable_container(
        css_styles=f"""
                    button {{
                            background-color: {COLORSCHEME["teal"]};
                            color: white;
                            border-color: white;
                        }}
                        """,
        key="teal_buttons",
    ):
        if st.button(
            "Finished setting up parameters?\n\nClick here to go to the model page",
            icon=":material/play_circle:",
        ):
            st.switch_page("model.py")
