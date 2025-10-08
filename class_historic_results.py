import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
import _processing_functions


class HistoricResults:
    """
    Manages historic results
    """

    def __init__(
        self,
        historical_data_folder_path,
        historic_rota_df_path,
        historic_callsign_df_path,
        historic_servicing_df_path,
    ):
        self.historical_data_path = historical_data_folder_path

        # Attendance dataframes
        self.historical_attendance_df = None

        # Missed calls
        self.historical_missed_calls_by_hour_df = None
        self.historical_missed_calls_by_quarter_and_hour_df = None

        # Jobs per month
        self.historical_jobs_per_month = None
        self.historical_monthly_totals_by_callsign = None

        # Jobs per day
        self.historical_jobs_per_day_per_callsign = None

        # Care categories
        self.care_cat_by_hour_historic = None

        # Activity durations
        self.historical_activity_durations_summary = None
        self.historical_activity_durations_breakdown = None

        # Utilisation (recorded)
        self.historical_monthly_resource_utilisation = None

        # Utilisation (calculated)
        self.historical_utilisation_df_complete = None
        self.historical_utilisation_df_summary = None

        # Historic rotas and scheduling details
        self.historic_rota_df_path = historic_rota_df_path
        self.historic_callsign_df_path = historic_callsign_df_path
        self.historic_servicing_df_path = historic_servicing_df_path

        self.historic_rota_df = pd.read_csv(self.historic_rota_df_path)
        self.historic_callsign_df = pd.read_csv(self.historic_callsign_df_path)
        self.historic_servicing_df = pd.read_csv(self.historic_servicing_df_path)

        # Calculated/simulated historical comparisons
        self.SIM_hist_params_missed_jobs = None
        self.SIM_hist_suboptimal_care_cat_summary = None
        self.SIM_hist_suboptimal_vehicle_type_summary = None
        self.SIM_historical_params_df = None
        self.SIM_hist_missed_care_cat_breakdown = None

        self.historical_monthly_totals_all_calls = None
        self.historical_monthly_totals_by_hour_of_day = None

        self.get_SIM_hist_params_missed_jobs_df()
        self.get_SIM_hist_missed_care_cat_breakdown_df()
        self.get_SIM_hist_suboptimal_care_cat_summary_df()
        self.get_SIM_hist_suboptimal_vehicle_type_summary_df()
        self.get_SIM_hist_params()

        self.get_historical_attendance_df()
        self.get_historical_missed_calls_by_hour()
        self.get_historical_monthly_resource_utilisation()
        self.get_historical_missed_calls_by_quarter_and_hour()
        self.get_care_cat_by_hour_historic()
        self.get_historical_jobs_per_day_per_callsign()
        self.get_historical_jobs_per_month()
        self.get_historical_activity_durations_summary()
        self.get_historical_monthly_totals_by_callsign()
        self.get_historical_monthly_totals_by_hour_of_day()

        self.get_historical_monthly_totals_all_calls()

        self.make_RWC_utilisation_dataframe()

    def get_historical_monthly_totals_by_hour_of_day(self):
        self.historical_monthly_totals_by_hour_of_day = pd.read_csv(
            f"{self.historical_data_path}/historical_monthly_totals_by_hour_of_day.csv"
        )

    def get_SIM_hist_params(self):
        self.SIM_hist_params = pd.read_csv(
            f"{self.historical_data_path}/calculated/run_params_used.csv"
        )

    def get_historical_monthly_totals_all_calls(self):
        self.historical_monthly_totals_all_calls = pd.read_csv(
            f"{self.historical_data_path}/historical_monthly_totals_all_calls.csv"
        )

        self.historical_monthly_totals_all_calls["month"] = pd.to_datetime(
            self.historical_monthly_totals_all_calls["month"], format="ISO8601"
        )

        self.historical_monthly_totals_all_calls["Month_Numeric"] = (
            self.historical_monthly_totals_all_calls["month"].apply(lambda x: x.month)
        )

        self.historical_monthly_totals_all_calls["Year_Numeric"] = (
            self.historical_monthly_totals_all_calls["month"].apply(lambda x: x.year)
        )

    def get_SIM_hist_params_missed_jobs_df(self):
        self.SIM_hist_params_missed_jobs = pd.read_csv(
            f"{self.historical_data_path}/calculated/SIM_hist_params_missed_jobs_care_cat_summary.csv"
        )

    def get_SIM_hist_missed_care_cat_breakdown_df(self):
        self.SIM_hist_missed_care_cat_breakdown = pd.read_csv(
            f"{self.historical_data_path}/calculated/SIM_hist_params_missed_jobs_care_cat_breakdown.csv"
        )

    def get_SIM_hist_suboptimal_care_cat_summary_df(self):
        self.SIM_hist_suboptimal_care_cat_summary = pd.read_csv(
            f"{self.historical_data_path}/calculated/SIM_hist_params_suboptimal_care_cat_sent_summary.csv"
        )

    def get_SIM_hist_suboptimal_vehicle_type_summary_df(self):
        self.SIM_hist_suboptimal_vehicle_type_summary = pd.read_csv(
            f"{self.historical_data_path}/calculated/SIM_hist_params_suboptimal_vehicle_type_sent_summary.csv"
        )

    def get_historical_attendance_df(self):
        self.historical_attendance_df = pd.read_csv(
            f"{self.historical_data_path}/historical_missed_calls_by_month.csv"
        )

        self.historical_attendance_df = self.historical_attendance_df.pivot(
            columns="callsign_group_simplified", index="month_start", values="count"
        ).reset_index()

        self.historical_attendance_df.rename(
            columns={
                "HEMS (helo or car) available and sent": "jobs_attended",
                "No HEMS available": "jobs_not_attended",
            },
            inplace=True,
        )

        self.historical_attendance_df["all_received_calls"] = (
            self.historical_attendance_df["jobs_attended"]
            + self.historical_attendance_df["jobs_not_attended"]
        )

        self.historical_attendance_df["perc_unattended_historical"] = (
            self.historical_attendance_df["jobs_not_attended"]
            / self.historical_attendance_df["all_received_calls"].round(2)
        )

    def get_historical_missed_calls_by_hour(self):
        self.historical_missed_calls_by_hour_df = pd.read_csv(
            f"{self.historical_data_path}/historical_missed_calls_by_hour.csv"
        )

    def get_historical_monthly_resource_utilisation(self):
        self.historical_monthly_resource_utilisation = pd.read_csv(
            f"{self.historical_data_path}/historical_monthly_resource_utilisation.csv"
        )

    def get_historical_missed_calls_by_quarter_and_hour(
        self,
    ):
        self.historical_missed_calls_by_quarter_and_hour_df = pd.read_csv(
            f"{self.historical_data_path}/historical_missed_calls_by_quarter_and_hour.csv"
        )

    def get_care_cat_by_hour_historic(self):
        self.care_cat_by_hour_historic = pd.read_csv(
            f"{self.historical_data_path}/historical_care_cat_counts.csv"
        )

        total_per_hour = self.care_cat_by_hour_historic.groupby("hour")[
            "count"
        ].transform("sum")

        # Add proportion column
        self.care_cat_by_hour_historic["proportion"] = (
            self.care_cat_by_hour_historic["count"] / total_per_hour
        )

    def get_historical_jobs_per_day_per_callsign(self):
        self.historical_jobs_per_day_per_callsign = pd.read_csv(
            f"{self.historical_data_path}/historical_jobs_per_day_per_callsign.csv"
        )

    def get_historical_jobs_per_month(self):
        self.historical_jobs_per_month = pd.read_csv(
            f"{self.historical_data_path}/historical_jobs_per_month.csv"
        )

        self.historical_jobs_per_month["month"] = pd.to_datetime(
            self.historical_jobs_per_month["month"], dayfirst=True
        )

    def get_historical_monthly_totals_by_callsign(self):
        self.historical_monthly_totals_by_callsign = pd.read_csv(
            f"{self.historical_data_path}/historical_monthly_totals_by_callsign.csv"
        )

    def get_historical_activity_durations_summary(self):
        self.historical_activity_durations_summary = pd.read_csv(
            f"{self.historical_data_path}/historical_median_time_of_activities_by_month_and_resource_type.csv",
            parse_dates=False,
        )

        # Parse month manually as more controllable
        self.historical_activity_durations_summary["month"] = pd.to_datetime(
            self.historical_activity_durations_summary["month"], format="%Y-%m-%d"
        )

    def get_historical_activity_durations_breakdown(self):
        self.historical_activity_durations_breakdown = pd.read_csv(
            f"{self.historical_data_path}/historical_job_durations_breakdown.csv"
        )

    def make_RWC_utilisation_dataframe(
        self,
    ):
        def calculate_theoretical_time(
            historical_df,
            rota_df,
            service_df,
            callsign_df,
            historical_monthly_resource_utilisation_df,
            long_format_df=True,
        ):
            # Pull in relevant rota, registration and servicing data
            rota_df = rota_df.merge(callsign_df, on="callsign")
            service_df = service_df.merge(callsign_df, on="registration")
            # print("==calculate_theoretical_time - rota_df after merging with callsign_df==")
            # print(rota_df)

            # Convert date columns to datetime format
            historical_monthly_resource_utilisation_df["month"] = pd.to_datetime(
                historical_monthly_resource_utilisation_df["month"]
            )

            # Create a dummy dataframe with every date in the range represented
            # We'll use this to make sure days with 0 activity get factored in to the calculations
            date_range = pd.date_range(
                start=historical_df["month"].min(),
                end=pd.offsets.MonthEnd().rollforward(historical_df["month"].max()),
                freq="D",
            )
            daily_df = pd.DataFrame({"date": date_range})

            # print("==historical_df==")
            # print(historical_df)

            service_df["service_start_date"] = pd.to_datetime(
                service_df["service_start_date"]
            )
            service_df["service_end_date"] = pd.to_datetime(
                service_df["service_end_date"]
            )

            def is_summer(date_obj):
                return current_date.month in [4, 5, 6, 7, 8, 9]

            # Initialize columns in df_availability for each unique callsign
            for callsign in rota_df["callsign"].unique():
                daily_df[callsign] = 0  # Initialize with 0 minutes

            daily_df = daily_df.set_index("date")

            # Iterate through each date in our availability dataframe
            for date_idx, current_date in enumerate(daily_df.index):
                is_current_date_summer = is_summer(current_date)

                # Iterate through each resource's rota entry
                for _, rota_entry in rota_df.iterrows():
                    callsign = rota_entry["callsign"]
                    start_hour_col = (
                        "summer_start" if is_current_date_summer else "winter_start"
                    )
                    end_hour_col = (
                        "summer_end" if is_current_date_summer else "winter_end"
                    )

                    start_hour = rota_entry[start_hour_col]
                    end_hour = rota_entry[end_hour_col]

                    # --- Calculate minutes for the current_date ---
                    minutes_for_callsign_on_date = 0

                    # Scenario 1: Shift is fully within one day (e.g., 7:00 to 19:00)
                    if start_hour < end_hour:
                        # Check if this shift is active on current_date (it always is in this logic,
                        # as we are calculating for the current_date based on its rota)
                        minutes_for_callsign_on_date = (end_hour - start_hour) * 60
                    # Scenario 2: Shift spans midnight (e.g., 19:00 to 02:00)
                    elif start_hour > end_hour:
                        # Part 1: Minutes from start_hour to midnight on current_date
                        minutes_today = (24 - start_hour) * 60
                        minutes_for_callsign_on_date += minutes_today

                        # Part 2: Minutes from midnight to end_hour on the *next* day
                        # These minutes need to be added to the *next day's* total for this callsign
                        if date_idx + 1 < len(
                            daily_df
                        ):  # Ensure there is a next day in our df
                            next_date = daily_df.index[date_idx + 1]
                            minutes_on_next_day = end_hour * 60
                            daily_df.loc[next_date, callsign] = (
                                daily_df.loc[next_date, callsign] + minutes_on_next_day
                            )

                    daily_df.loc[current_date, callsign] += minutes_for_callsign_on_date

            theoretical_availability = daily_df.copy().reset_index()
            theoretical_availability["month"] = theoretical_availability[
                "date"
            ].dt.strftime("%Y-%m-01")
            theoretical_availability = theoretical_availability.drop(columns=["date"])
            theoretical_availability = theoretical_availability.set_index("month")
            theoretical_availability = theoretical_availability.groupby("month").sum()
            theoretical_availability = theoretical_availability.reset_index()

            # print("==_utilisation_result_calculation.py - make_RWC_utilisation_dataframe - theoretical availability df==")
            # print(theoretical_availability_df)

            theoretical_availability.to_csv(
                "historical_data/calculated/theoretical_availability_historical.csv",
                index=False,
            )

            if long_format_df:
                theoretical_availability_df = theoretical_availability.melt(
                    id_vars="month"
                ).rename(
                    columns={
                        "value": "theoretical_availability",
                        "variable": "callsign",
                    }
                )

                theoretical_availability_df["theoretical_availability"] = (
                    theoretical_availability_df["theoretical_availability"].astype(
                        "float"
                    )
                )

                theoretical_availability_df = theoretical_availability_df.fillna(0.0)

            return theoretical_availability_df

        theoretical_availability_df = calculate_theoretical_time(
            historical_df=self.historical_monthly_resource_utilisation,
            rota_df=self.historic_rota_df,
            callsign_df=self.historic_callsign_df,
            service_df=self.historic_servicing_df,
            historical_monthly_resource_utilisation_df=self.historical_monthly_resource_utilisation,
            long_format_df=True,
        )

        # print("==theoretical_availability_df==")
        # print(theoretical_availability_df)
        theoretical_availability_df["month"] = pd.to_datetime(
            theoretical_availability_df["month"]
        )

        # theoretical_availability_df.to_csv("historical_data/calculated/theoretical_availability_historical.csv")

        historical_utilisation_df_times = (
            self.historical_monthly_resource_utilisation.set_index("month")
            .filter(like="total_time")
            .reset_index()
        )

        historical_utilisation_df_times.columns = [
            x.replace("total_time_", "")
            for x in historical_utilisation_df_times.columns
        ]

        historical_utilisation_df_times = historical_utilisation_df_times.melt(
            id_vars="month"
        ).rename(columns={"value": "usage", "variable": "callsign"})

        historical_utilisation_df_times = historical_utilisation_df_times.fillna(0)

        self.historical_utilisation_df_complete = pd.merge(
            left=historical_utilisation_df_times,
            right=theoretical_availability_df,
            on=["callsign", "month"],
            how="left",
        )

        self.historical_utilisation_df_complete["percentage_utilisation"] = (
            self.historical_utilisation_df_complete["usage"]
            / self.historical_utilisation_df_complete["theoretical_availability"]
        )

        self.historical_utilisation_df_complete["percentage_utilisation_display"] = (
            self.historical_utilisation_df_complete["percentage_utilisation"].apply(
                lambda x: f"{x:.1%}"
            )
        )

        self.historical_utilisation_df_complete.to_csv(
            "historical_data/calculated/complete_utilisation_historical.csv"
        )

        self.historical_utilisation_df_summary = (
            self.historical_utilisation_df_complete.groupby("callsign")[
                "percentage_utilisation"
            ].agg(["min", "max", "mean", "median"])
            * 100
        ).round(1)

        self.historical_utilisation_df_summary.to_csv(
            "historical_data/calculated/complete_utilisation_historical_summary.csv"
        )

    def PLOT_historical_missed_jobs_data(self, format="stacked_bar") -> Figure:
        if self.historical_attendance_df is None:
            try:
                self.get_historical_attendance_df()
            except FileNotFoundError:
                raise (
                    "Historical attendance df not found. Please run the method get_historical_attendance_df(),"
                    "passiing in the path to the historical data on missed calls per month"
                )
        if format == "stacked_bar":
            return px.bar(
                self.historical_attendance_df[
                    ["month", "jobs_not_attended", "jobs_attended"]
                ].melt(id_vars="month"),
                x="month",
                y="value",
                color="variable",
            )

        elif format == "line_not_attended_count":
            return px.line(
                self.historical_attendance_df, x="month", y="jobs_not_attended"
            )

        elif format == "line_not_attended_perc":
            return px.line(
                self.historical_attendance_df, x="month", y="perc_unattended_historical"
            )

        elif format == "string":
            # This approach can distort the result by giving more weight to months with higher numbers of calls
            # However, for system-level performance, which is what we care about here, it's a reasonable option
            all_received_calls_period = self.historical_attendance_df[
                "all_received_calls"
            ].sum()
            all_attended_jobs_period = self.historical_attendance_df[
                "jobs_attended"
            ].sum()
            return (
                (all_received_calls_period - all_attended_jobs_period)
                / all_received_calls_period
            ) * 100

            # Alternative is to take the mean of means
            # return full_jobs_df['perc_unattended_historical'].mean()*100

        else:
            # Melt the DataFrame to long format
            df_melted = self.historical_attendance_df[
                ["month", "jobs_not_attended", "jobs_attended"]
            ].melt(id_vars="month")

            # Calculate proportions per month
            df_melted["proportion"] = df_melted.groupby("month")["value"].transform(
                lambda x: x / x.sum()
            )

            # Plot proportions
            fig = px.bar(
                df_melted,
                x="month",
                y="proportion",
                color="variable",
                text="value",  # Optional: to still show raw values on hover
            )

            fig.update_layout(
                barmode="stack", yaxis_tickformat=".0%", yaxis_title="Proportion"
            )
            fig.show()

    def PLOT_care_cat_counts_historic(self, show_proportions=False) -> Figure:
        title = "Care Category of calls in historical data by hour of day with EC/CC/Regular - Heli Benefit/Regular"

        if not show_proportions:
            fig = px.bar(
                self.care_cat_by_hour_historic,
                x="hour",
                y="count",
                color="care_category",
                title=title,
                category_orders={
                    "care_category": [
                        "CC",
                        "EC",
                        "REG - helicopter benefit",
                        "REG",
                        "Unknown - DAA resource did not attend",
                    ]
                },
            )
        else:
            fig = px.bar(
                self.care_cat_by_hour_historic,
                x="hour",
                y="proportion",
                color="care_category",
                title=title,
                category_orders={
                    "care_category": [
                        "CC",
                        "EC",
                        "REG - helicopter benefit",
                        "REG",
                        "Unknown - DAA resource did not attend",
                    ]
                },
            )

        fig.update_layout(xaxis=dict(dtick=1))

        return fig

    def PLOT_RWC_utilisation(self) -> Figure:
        fig = px.box(
            self.historical_utilisation_df_complete,
            x="percentage_utilisation",
            y="callsign",
        )

        return fig

    def RETURN_missed_jobs_fig(self, care_category, what="average") -> str:
        row = self.SIM_hist_params_missed_jobs[
            (self.SIM_hist_params_missed_jobs["care_cat"] == care_category)
            & (self.SIM_hist_params_missed_jobs["time_type"] == "No Resource Available")
        ]
        if what == "average":
            return row["jobs_per_year_average"].values[0]
        elif what == "min":
            return row["jobs_per_year_min"].values[0]
        elif what == "max":
            return row["jobs_per_year_max"].values[0]

    def RETURN_hist_util_fig(self, callsign="H70", average="mean") -> str:
        try:
            return self.historical_utilisation_df_summary[
                self.historical_utilisation_df_summary.index == callsign
            ][average].values[0]
        except IndexError:
            return f"Error returning value for callsign {callsign}. Available callsigns are {self.historical_utilisation_df_summary.index.unique()}"

    def RETURN_prediction_cc_patients_sent_ec_resource(self) -> tuple:
        row_of_interest = self.SIM_hist_suboptimal_care_cat_summary[
            (self.SIM_hist_suboptimal_care_cat_summary["hems_res_category"] != "CC")
            & (self.SIM_hist_suboptimal_care_cat_summary["care_cat"] == "CC")
        ]

        run_duration_days = float(
            _processing_functions.get_param("sim_duration_days", self.SIM_hist_params)
        )

        return (
            (row_of_interest["mean"].values[0] / run_duration_days) * 365,
            (row_of_interest["min"].values[0] / run_duration_days) * 365,
            (row_of_interest["max"].values[0] / run_duration_days) * 365,
        )

    def RETURN_prediction_heli_benefit_patients_sent_car(self) -> tuple:
        row_of_interest = self.SIM_hist_suboptimal_vehicle_type_summary[
            (self.SIM_hist_suboptimal_vehicle_type_summary["vehicle_type"] == "car")
            & (self.SIM_hist_suboptimal_vehicle_type_summary["heli_benefit"] == "y")
        ]

        run_duration_days = float(
            _processing_functions.get_param("sim_duration_days", self.SIM_hist_params)
        )

        return (
            (row_of_interest["mean"].values[0] / run_duration_days) * 365,
            (row_of_interest["min"].values[0] / run_duration_days) * 365,
            (row_of_interest["max"].values[0] / run_duration_days) * 365,
        )
