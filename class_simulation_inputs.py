import pandas as pd


class SimulationInputs:
    def __init__(
        self,
        params_df_path="../data/run_params_used.csv",
        rota_path="../actual_data/HEMS_ROTA.csv",
        service_path="../data/service_dates.csv",
        callsign_path="../actual_data/callsign_registration_lookup.csv",
        rota_times="../actual_data/rota_start_end_months.csv",
    ):
        self.params_df_path = params_df_path
        self.rota_path = rota_path
        self.service_path = service_path
        self.callsign_path = callsign_path
        self.rota_times = rota_times

        self.params_df = pd.read_csv(self.params_df_path)
        self.rota_df = pd.read_csv(self.rota_path)
        self.service_dates_df = pd.read_csv(self.service_path)
        self.callsign_registration_lookup_df = pd.read_csv(self.callsign_path)
        self.rota_start_end_months_df = pd.read_csv(self.rota_times)
