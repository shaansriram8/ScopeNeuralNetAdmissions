import pandas as pd
import re
from datetime import datetime

def clean_scope_data(df):
    """
    Clean and standardize SCOPE Consulting application data.
    
    Parameters:
    df (pandas.DataFrame): Raw application data
    
    Returns:
    pandas.DataFrame: Cleaned and standardized data
    """
    # Create a copy to avoid modifying original data
    cleaned_df = df.copy()
    
    # Rename columns to be more consistent
    column_mapping = {
        'Timestamp': 'submission_timestamp',
        'Year in School:': 'year_in_school',
        'Expected Graduation Date (Month and Year):': 'expected_graduation',
        'Major:': 'major',
        'Minor (if applicable)': 'minor',
        'Have you applied to scope before': 'previous_applicant',
        'Are you willing to commit at least 4 hours a week to the club (including general meetings and weekly project team meetings)?': 'willing_to_commit',
        'Do you have any conflicts on Tuesdays from 6-7 pm?': 'tuesday_conflicts',
        'Do you plan to be on campus this semester?': 'on_campus',
        'List all the extracurricular activities you plan on being involved in during this semester. Please include estimated time commitment per week.': 'extracurricular_activities',
        'Why are you interested in joining SCOPE Consulting? (100 - 200 words)': 'interest_reason',
        'What skills, expertise and knowledge will you bring to SCOPE Consulting and your team?': 'skills_expertise',
        'What do you hope to gain from this experience?': 'expected_gains',
        'How did you hear about us? (Check all that apply; if you heard from a friend please check other and enter their name)': 'referral_source',
        'Is there anything else you would like SCOPE board to know?': 'additional_info'
    }
    cleaned_df.rename(columns=column_mapping, inplace=True)
    
    # Standardize timestamps
    cleaned_df['submission_timestamp'] = pd.to_datetime(cleaned_df['submission_timestamp'])
    
    # Standardize yes/no responses
    for col in ['willing_to_commit', 'tuesday_conflicts', 'on_campus']:
        cleaned_df[col] = cleaned_df[col].str.lower()
        cleaned_df[col] = cleaned_df[col].map({'yes': True, 'no': False, 'maybe': None})
    
    # Clean previous applicant responses
    cleaned_df['previous_applicant'] = cleaned_df['previous_applicant'].fillna('No')
    cleaned_df['previous_applicant'] = cleaned_df['previous_applicant'].str.contains('Yes', case=False, na=False)
    
    # Standardize year in school
    year_mapping = {
        'Freshman': 1,
        'Sophomore': 2,
        'Junior': 3,
        'Senior': 4,
        'Graduate Student': 5,
        '5th Year': 5
    }
    cleaned_df['year_in_school_numeric'] = cleaned_df['year_in_school'].map(year_mapping)
    
    # Extract expected graduation month and year separately
    def extract_graduation_date(date_str):
        try:
            return pd.to_datetime(date_str, format='%B %Y')
        except:
            try:
                return pd.to_datetime(date_str)
            except:
                return None
    
    cleaned_df['expected_graduation'] = cleaned_df['expected_graduation'].apply(extract_graduation_date)
    cleaned_df['graduation_year'] = cleaned_df['expected_graduation'].dt.year
    cleaned_df['graduation_month'] = cleaned_df['expected_graduation'].dt.month
    
    # Clean up major/minor fields
    cleaned_df['major'] = cleaned_df['major'].fillna('')
    cleaned_df['minor'] = cleaned_df['minor'].fillna('')
    cleaned_df['major'] = cleaned_df['major'].str.strip()
    cleaned_df['minor'] = cleaned_df['minor'].str.strip()
    
    # Extract time commitments from extracurricular activities
    def extract_hours(text):
        if pd.isna(text):
            return 0
        hours = re.findall(r'(\d+(?:\.\d+)?)\s*(?:hour|hr|h)s?(?:\s*\/\s*|\s+per\s+)?(?:week|wk|w)?', text.lower())
        return sum(float(h) for h in hours) if hours else 0
    
    cleaned_df['extracurricular_hours'] = cleaned_df['extracurricular_activities'].apply(extract_hours)
    
    # Clean up text fields
    text_columns = ['interest_reason', 'skills_expertise', 'expected_gains', 'additional_info']
    for col in text_columns:
        cleaned_df[col] = cleaned_df[col].fillna('')
        cleaned_df[col] = cleaned_df[col].str.strip()
        cleaned_df[col] = cleaned_df[col].replace(r'\s+', ' ', regex=True)
    
    # Standardize referral sources
    cleaned_df['referral_source'] = cleaned_df['referral_source'].fillna('')
    cleaned_df['referral_source'] = cleaned_df['referral_source'].str.strip()
    
    return cleaned_df

#Example usage:
df = pd.read_csv('scope_applications.csv')
cleaned_df = clean_scope_data(df)

with open('cleaned_scope_data.txt', 'w', encoding='utf-8') as f:
    for index, row in cleaned_df.iterrows():
        # Write each row as a formatted string
        f.write(f"Application {index + 1}\n")
        for column in cleaned_df.columns:
            if str(row[column]).strip():  # Only write non-empty values
                f.write(f"{column}: {row[column]}\n")
        f.write("\n---\n\n")  # Add separator between applications

print("Data has been written to 'cleaned_scope_data.txt'")