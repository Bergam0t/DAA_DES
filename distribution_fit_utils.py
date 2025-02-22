from csv import QUOTE_ALL
import numpy as np
import pandas as pd
import json
from fitter import Fitter, get_common_distributions
from datetime import timedelta
from utils import Utils

class DistributionFitUtils():
    """
        # The DistributionFitUtils classa

        This class will import a CSV, undertake some light
        wrangling and then determine distributions and probabilities required
        for the Discrete Event Simulation

        example usage: 
            my_data = DistributionFitUtils('data/my_data.csv')
            my_data.import_and_wrangle()
  
    """

    def __init__(self, file_path: str, calculate_school_holidays = False, school_holidays_years = 5):
       
        self.file_path = file_path
        self.df = pd.DataFrame()
        
        # The number of additional years of school holidays
        # that will be calculated over that maximum date in the provided dataset
        self.school_holidays_years = school_holidays_years 
        self.calculate_school_holidays = calculate_school_holidays 
        
        self.times_to_fit = [
            {"hems_result": "Patient Treated (not conveyed)", 
            "times_to_fit" : ['time_allocation', 'time_mobile', 'time_to_scene', 'time_on_scene', 'time_to_clear']},
            {"hems_result": "Patient Conveyed" , "times_to_fit" : ['time_allocation', 'time_mobile', 'time_to_scene', 'time_on_scene', 'time_to_hospital', 'time_to_clear']},
            {"hems_result": "Stand Down Before Mobile" , "times_to_fit" : ['time_allocation', 'time_to_clear']},
            {"hems_result": "Stand Down En Route" , "times_to_fit" : ['time_allocation', 'time_mobile', 'time_to_clear']},
            {"hems_result": "Landed but no patient contact" , "times_to_fit" : ['time_allocation', 'time_mobile', 'time_to_scene', 'time_on_scene', 'time_to_clear']},
        ]

        self.sim_tools_distr_plus = [
            "poisson",
            "bernoulli",
            "triang",
            "erlang",
            "weibull_min",
            "expon_weib",
            "betabinom",
            "pearson3",
        ] + get_common_distributions()

    def getBestFit(self, q_times, distr=get_common_distributions(), show_summary=False):
        """
        
            Convenience function for Fitter. 
            Returns model and parameters that is considered
            the 'best fit'.

            TODO: Determine how Fitter works this out
        
        """

        if(q_times.size > 0):
            if(len(distr) > 0):
                f = Fitter(q_times, timeout=60, distributions=distr)
            else:  
                f = Fitter(q_times, timeout=60)
            f.fit()
            if show_summary == True:
                f.summary()
            return f.get_best()
        else:
            return {}


    def import_and_wrangle(self):
        """
        
            Function to import CSV, add additional columns that are required
            and then sequentially execute other class functions to generate
            the probabilities and distributions required for the DES.

            TODO: Additional logic is required to check the imported CSV
            for missing values, incorrect columns names etc.
        
        """

        try:
            df = pd.read_csv(self.file_path, quoting=QUOTE_ALL)
            self.df = df
            
            # Perhaps run some kind of checking function here.

        except FileNotFoundError:
            print(f"Cannot locate that file")

        # If everything is okay, crack on...
        self.df['inc_date'] = pd.to_datetime(self.df['inc_date'])
        self.df['date_only'] = pd.to_datetime(df['inc_date'].dt.date)
        self.df['hour'] = self.df['inc_date'].dt.hour                      # Hour of the day
        self.df['day_of_week'] = self.df['inc_date'].dt.day_name()         # Day of the week (e.g., Monday)
        self.df['month'] = self.df['inc_date'].dt.month
        self.df['quarter'] = self.df['inc_date'].dt.quarter   

        # This will be needed for other datasets, but has already been computed for DAA
        #self.df['ampds_card'] = self.df['ampds_code'].str[:2]

        #get proportions of AMPDS card by hour of day
        self.hour_by_ampds_card_probs()

        # Determine 'best' distributions for time-based data
        self.activity_time_distributions()

        # Calculate probability patient will be female based on AMPDS card
        self.sex_by_ampds_card_probs()

        # Determine 'best' distributions for age ranges straitifed by AMPDS card
        self.age_distributions()

        # Calculate the mean inter-arrival times stratified by yearly quarter and hour of day
        self.inter_arrival_times()

        # Alternaitve approach to IA times. Start with probabilty of call at given hour stratified by quarter
        self.hourly_arrival_by_qtr_probs()

        # Calculates the mean and standard deviaion of the number of incidents per day stratified by quarter
        self.incidents_per_day()

        # Calculate probabily of callsign being allocated to a job based on AMPDS card and hour of day
        self.callsign_group_by_ampds_card_and_hour_probs()

        # Calculate probabily of HEMS result being allocated to a job based on callsign and hour of day
        self.hems_result_by_callsign_group_and_vehicle_type_probs()

        # Calculate probability of a specific patient outcome being allocated to a job based on HEMS result and callsign
        self.pt_outcome_by_hems_result_probs()

        # Calculate probability of a particular vehicle type based on callsign group and month of year
        self.vehicle_type_by_month_probs()

        # Calculate school holidays since servicing schedules typically avoid these dates
        if self.calculate_school_holidays:
            self.school_holidays()
            

    def hour_by_ampds_card_probs(self):
        """
        
            Calculates the proportions of calls that are triaged with 
            a specific AMPDS card. This is stratified by hour of day

            TODO: Determine whether this should also be stratified by yearly quarter
        
        """
        category_counts = self.df.groupby(['hour', 'ampds_card']).size().reset_index(name='count')
        total_counts = category_counts.groupby('hour')['count'].transform('sum')
        category_counts['proportion'] = round(category_counts['count'] / total_counts, 4)

        #category_counts['ampds_card'] = category_counts['ampds_card'].apply(lambda x: str(x).zfill(2))

        category_counts.to_csv('distribution_data/hour_by_ampds_card_probs.csv', mode="w+")


    def sex_by_ampds_card_probs(self):
        """
        
            Calculates the probability that the patient will be female
            stratified by AMPDS card.
        
        """
        age_df = self.df
        category_counts = age_df.groupby(['ampds_card', 'sex']).size().reset_index(name='count')
        total_counts = category_counts.groupby('ampds_card')['count'].transform('sum')
        category_counts['proportion'] = round(category_counts['count'] / total_counts, 3)

        category_counts[category_counts['sex'] =='Female'].to_csv('distribution_data/sex_by_ampds_card_probs.csv', mode="w+")


    def activity_time_distributions(self):
        """
        
            Determine the 'best' distribution for each phase of a call
            i.e. Allocation time, Mobilisation time, Time to scene
            Time on scene, Travel time to hospital and handover, Time to clear.
            Not all times will apply to all cases, so the class 'times_to_fit'
            variable is a list of dictionaries, which contains the times to fit
            
            The data is currently stratitied by HEMS_result and vehicle type fields.
        
        """
       
        vehicle_type = self.df['vehicle_type'].unique()

        # We'll need to make sure that where a distribution is missing that the time is set to 0 in the model.
        # Probably easier than complicated logic to determine what times should be available based on hems_result

        final_distr = []

        for row in self.times_to_fit:
            print(row)
            for ttf in row['times_to_fit']:
                for vt in vehicle_type:
                    #print(f"HEMS result is {row['hems_result']} cs is {cs} and times_to_fit is {ttf} and patient outcome {pto}")

                    # This line might not be required if data quality is determined when importing the data
                    max_time = 20 if ttf == "time_mobile" else 120
                    fit_times = self.df[
                        (self.df.vehicle_type == vt) & 
                        (self.df[ttf] > 0) & 
                        (self.df[ttf] < max_time) & 
                        (self.df.hems_result == row['hems_result'])
                    ][ttf]
                    #print(fit_times[:10])
                    best_fit = self.getBestFit(fit_times, distr=self.sim_tools_distr_plus)
                    return_dict = { "vehicle_type": vt, "time_type" : ttf, "best_fit": best_fit, "hems_result": row['hems_result'], "n": len(fit_times)}
                    #print(return_dict)
                    final_distr.append(return_dict)

        with open('distribution_data/activity_time_distributions.txt', 'w+') as convert_file:
            convert_file.write(json.dumps(final_distr))
        convert_file.close()


    def age_distributions(self):
        """
        
            Determine the 'best' distribution for age stratified by
            AMPDS card
        
        """

        age_distr = []

        age_df = self.df[["age", "ampds_card"]].dropna()
        ampds_cards = age_df['ampds_card'].unique()
        print(ampds_cards)

        for card in ampds_cards:
            fit_ages = age_df[age_df['ampds_card'] == card]['age']
            best_fit = self.getBestFit(fit_ages, distr=self.sim_tools_distr_plus)
            return_dict = { "ampds_card": str(card), "best_fit": best_fit, "n": len(fit_ages)}
            age_distr.append(return_dict)

        with open('distribution_data/age_distributions.txt', 'w+') as convert_file:
            convert_file.write(json.dumps(age_distr))
        convert_file.close()


    def inter_arrival_times(self):
        """
        
            Calculate the mean inter-arrival times for patients
            stratified by hour, and and yearly quarter
        
        """

        ia_df = self.df[['date_only', 'quarter', 'hour']].dropna()

        count_df = ia_df.groupby(['hour', 'date_only', 'quarter']).size().reset_index(name='n')

        ia_times_df = (
            count_df.groupby(['hour', 'quarter'])
            .agg(
                # max_arrivals_per_hour=('n', lambda x: round(60 / np.max(x), 3)),
                # min_arrivals_per_hour=('n', lambda x: round(60 / np.min(x),3)),
                mean_cases=('n', lambda x: round(x.mean(), 1)),
                # sd_cases=('n', lambda x: round(x.std(), 3)), 
                mean_iat=('n', lambda x: 60 / x.mean())
                # n=('n', 'size')
            )
            .reset_index()
        )
        # Additional column for NSPPThinning
        ia_times_df['t'] = ia_times_df['hour']
        ia_times_df['arrival_rate'] = ia_times_df['mean_iat'].apply(lambda x: 1/x)

        ia_times_df.to_csv('distribution_data/inter_arrival_times.csv', mode='w+')


    def incidents_per_day(self):
        """
        
            Determine the best fitting distribution for incidents per
            day stratified by quarter
        
        """

        inc_df = self.df[['inc_date', 'date_only', 'quarter']].dropna()\
            .drop_duplicates(subset="inc_date", keep="first")
        
        #print(inc_df.shape)
        
        jpd_df = inc_df.groupby(['date_only', 'quarter']).size().reset_index(name = 'jobs_per_day')

        quarters = jpd_df['quarter'].unique()

        jpd_distr = []

        for q in quarters:
            fit_quarter = jpd_df[jpd_df['quarter'] == q]['jobs_per_day']
            best_fit = self.getBestFit(fit_quarter, distr=self.sim_tools_distr_plus )
            return_dict = { "quarter": int(q), "best_fit": best_fit, "n": len(fit_quarter)}
            jpd_distr.append(return_dict)
            
        with open('distribution_data/inc_per_day_distributions.txt', 'w+') as convert_file:
            convert_file.write(json.dumps(jpd_distr))
        convert_file.close()


    def hourly_arrival_by_qtr_probs(self):
        """
        
            Calculates the proportions of calls arriving in any given hour
            stratified by yearly quarter
        
        """

        ia_df = self.df[['quarter', 'hour']].dropna()

        hourly_counts = ia_df.groupby(['hour', 'quarter']).size().reset_index(name='count')
        total_counts = hourly_counts.groupby(['quarter'])['count'].transform('sum')
        hourly_counts['proportion'] = round(hourly_counts['count'] / total_counts, 4)

        hourly_counts.sort_values(by=['quarter', 'hour']).to_csv('distribution_data/hourly_arrival_by_qtr_probs.csv', mode="w+")


    def callsign_group_by_ampds_card_and_hour_probs(self):
        """
        
            Calculates the probabilty of a specific callsign being allocated to
            a call based on the AMPDS card category and hour of day
        
        """
        callsign_counts = self.df.groupby(['ampds_card', 'hour', 'callsign_group']).size().reset_index(name='count')

        total_counts = callsign_counts.groupby(['ampds_card', 'hour'])['count'].transform('sum')
        callsign_counts['proportion'] = round(callsign_counts['count'] / total_counts, 4)

        callsign_counts.to_csv('distribution_data/callsign_group_by_ampds_card_and_hour_probs.csv', mode = "w+")


    def vehicle_type_by_month_probs(self):
        """
        
            Calculates the probabilty of a car/helicopter being allocated to
            a call based on the callsign group and month of the year
        
        """
        callsign_counts = self.df.groupby(['callsign_group', 'month', 'vehicle_type']).size().reset_index(name='count')

        total_counts = callsign_counts.groupby(['callsign_group', 'month'])['count'].transform('sum')
        callsign_counts['proportion'] = round(callsign_counts['count'] / total_counts, 4)

        callsign_counts.to_csv('distribution_data/vehicle_type_by_month_probs.csv', mode = "w+")


    def hems_result_by_callsign_group_and_vehicle_type_probs(self):
        """
        
            Calculates the probabilty of a specific HEMS result being allocated to
            a call based on the callsign group and hour of day

            TODO: These probability calculation functions could probably be refactored into a single
            function and just specify columns and output name
        
        """
        hems_counts = self.df.groupby(['hems_result', 'callsign_group', 'vehicle_type']).size().reset_index(name='count')

        total_counts = hems_counts.groupby(['callsign_group', 'vehicle_type'])['count'].transform('sum')
        hems_counts['proportion'] = round(hems_counts['count'] / total_counts, 4)

        hems_counts.to_csv('distribution_data/hems_result_by_callsign_group_and_vehicle_type_probs.csv', mode = "w+")


    def pt_outcome_by_hems_result_probs(self):
        """
        
            Calculates the probabilty of a specific patient outcome based on HEMS result
        
        """
        po_counts = self.df.groupby(['pt_outcome', 'hems_result']).size().reset_index(name='count')

        total_counts = po_counts.groupby(['hems_result'])['count'].transform('sum')
        po_counts['proportion'] = round(po_counts['count'] / total_counts, 4)

        po_counts.to_csv('distribution_data/pt_outcome_by_hems_result_probs.csv', mode = "w+")

    def school_holidays(self):

        min_date = self.df.inc_date.min()
        max_date = self.df.inc_date.max() + timedelta(weeks = (52 * self.school_holidays_years ))

        u = Utils()

        years_of_holidays_list = u.years_between(min_date, max_date)

        sh = pd.DataFrame(columns=['start_date', 'end_date'])

        for i, year in enumerate(years_of_holidays_list):
            tmp = u.calculate_term_holidays(year)

            if i == 0:
                sh = tmp
            else:
                sh = pd.concat([sh, tmp])

        sh.to_csv('actual_data/school_holidays.csv', index = False)

if __name__ == "__main__":
    from distribution_fit_utils import DistributionFitUtils
    test = DistributionFitUtils('external_data/clean_daa_import.csv', True)
    #test = DistributionFitUtils('external_data/clean_daa_import-2023.csv')
    test.import_and_wrangle()

# Testing ----------
# python distribution_fit_utils.py
