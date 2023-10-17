import pandas as pd
import numpy as np
import joblib

from expected_disposal_model.data_preparation.preprocessing import convert_chains_to_schema, filter_disposals, create_labels
from expected_disposal_model.config import raw_file_path, preprocessor_file_path, model_v1_file_path, scored_disposal_output

def predict_model(raw_file_path, preprocessor_file_path, model_file_path, output_path):

    # Load data
    chains = pd.read_csv(raw_file_path)
    print("Chain data loaded.")

    # Processing
    preproc = joblib.load(preprocessor_file_path)
    chain_features = preproc.transform(chains)
    
    schema_chains = convert_chains_to_schema(chains)
    disposals = filter_disposals(schema_chains)
    
    labels = create_labels(schema_chains)
    
    schema_chains = pd.concat([disposals, chain_features, labels], axis=1)
    
    print("Preprocessing.. Complete.")

    # Load model
    exp_disposal_model = joblib.load(model_file_path)
    
    # Scoring Model
    schema_chains['xDisposal'] = exp_disposal_model.predict(chain_features)
    print("Scoring.. complete.")
    
    # Merge back to chains
    chains = chains.merge(schema_chains, how = "left", left_on=['Match_ID', 'Chain_Number', 'Order'], right_on=['match_id', 'chain_number', 'order'])

    # Export data
    chains.to_csv(output_path, index=False)
    print("Exporting.. complete.")

if __name__ == "__main__":
    
    predict_model(raw_file_path,
                  preprocessor_file_path,
                  model_v1_file_path,
                  scored_disposal_output
                  )
    
    
    
    
    
    