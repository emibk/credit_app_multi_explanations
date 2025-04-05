from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import os
import pickle
import pandas as pd
import dice_ml
from dice_ml.utils import helpers
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fairlearn.metrics import demographic_parity_ratio
from fairlearn.metrics import equalized_odds_ratio
import shap
from fairlearn.metrics import MetricFrame, selection_rate, count, false_positive_rate, false_negative_rate
from catboost import Pool
# Create your views here.

def index(request):
    return render(request, "index.html")

MODEL_PATH1 = os.path.join(os.path.dirname(__file__), "catboost_model_1.pkl")

with open(MODEL_PATH1, "rb") as f:
    model1 = pickle.load(f)



def check_loan(request):
    prediction = None
    exp_df = pd.DataFrame()
    if request.method == "POST":
        try:
            Account_status = request.POST["Account_status"]
            Months =int(request.POST["Months"])
            Credit_history = request.POST["Credit_history"]
            Purpose = request.POST["Purpose"]
            Credit_amount = int(request.POST["Credit_amount"])
            Savings = request.POST["Savings"]
            Employment = request.POST["Employment"]
            Installment_rate = int(request.POST["Installment_rate"])
            Personal_status = request.POST["Personal_status"]
            Other_debtors = request.POST["Other_debtors"]
            Residence = int(request.POST["Residence"])
            Property = request.POST["Property"]
            Age = int(request.POST["Age"])
            Other_installments = request.POST["Other_installments"]
            Housing = request.POST["Housing"]
            Number_credits = int(request.POST["Number_credits"])
            Job = request.POST["Job"]
            Number_dependents = int(request.POST["Number_dependents"])
            Telephone = request.POST["Telephone"]
            Foreign_worker = request.POST["Foreign_worker"]

            input_data = {
                'Account_status': Account_status,
                'Months': Months,
                'Credit_history': Credit_history,
                'Purpose': Purpose,
                'Credit_amount': Credit_amount,
                'Savings': Savings,         
                'Employment': Employment,
                'Installment_rate': Installment_rate,
                'Personal_status': Personal_status,
                'Other_debtors': Other_debtors,
                'Residence': Residence,
                'Property': Property,
                'Age': Age,
                'Other_installments': Other_installments,
                'Housing': Housing,
                'Number_credits': Number_credits,
                'Job': Job,
                'Number_dependents': Number_dependents,
                'Telephone': Telephone,
                'Foreign_worker': Foreign_worker
            }
            # print(request.POST) 

            # print(input_data)  # Debugging step
            # print("Before creating DataFrame")  # Debugging step
            input_df = pd.DataFrame(input_data, index=[0])

            # print(input_df)
            prediction = model1.predict(input_df)
            # print("Prediction made")  # Debugging step
            # print(prediction)
            '''
            prediction = "Creditworthy" if model_prediction[0] == 1 else "Non-Creditworthy"
            data = dice_ml.Data(features={'Account_status': ['A11', 'A12', 'A13', 'A14'],
                            'Months': [1, 90],
                            'Credit_history': ['A30', 'A31', 'A32', 'A33', 'A34'],
                            'Purpose': ['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A48', 'A49', 'A410'],
                            'Credit_amount': [10, 30000],
                            'Savings': ['A61', 'A62', 'A63', 'A64', 'A65'],
                            'Employment': ['A71', 'A72', 'A73', 'A74', 'A75'],
                            'Installment_rate': [1, 4],
                            'Personal_status': ['A91', 'A92', 'A93', 'A94'],
                            'Other_debtors': ['A101', 'A102', 'A103'],
                            'Residence': [1, 4],
                            'Property': ['A121', 'A122', 'A123', 'A124'],
                            'Age': [19, 90],
                            'Other_installments': ['A141', 'A142', 'A143'],
                            'Housing': ['A151', 'A152', 'A153'],
                            'Number_credits': [1, 4],
                            'Job': ['A171', 'A172', 'A173', 'A174'],
                            'Number_dependents': [1, 5],
                            'Telephone': ['A191', 'A192'],
                            'Foreign_worker': ['A201', 'A202']    
                           },
                 outcome_name='target')
            model_exp = dice_ml.Model(model=model1, backend="sklearn", model_type='classifier')
            exp = dice_ml.Dice(data, model_exp, method="random")
            exp_user = exp.generate_counterfactuals(input_df, total_CFs=2, desired_class="opposite")
            # exp_user.visualize_as_dataframe(show_only_changes=True)
            exp_df = exp_user.cf_examples_list[0].final_cfs_df
            print(exp_df)
            G = create_explanation_graph(exp_df.iloc[0], prediction, input_df)
            # pos = nx.spring_layout(G)
            # nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10)
            # edge_labels = nx.get_edge_attributes(G, 'relation')
            # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            # plt.show()
            '''
            # nle = generate_nle(G, "Explanation")
            # print(nle)

            #print(identify_changes(exp_df.iloc[0], input_df))
            request.session['prediction'] = str(prediction[0])
            request.session['loan_data'] = input_data
            
            context = ""
            if prediction[0] == 1:
                context = {'prediction': "Creditworthy"}
            else:
                context = {'prediction': "Non-Creditworthy"}
            
            return render(request, "check_loan.html", context)

            
        except Exception as e:
            print(e)
            return HttpResponse("Invalid input")
    



    return render(request, "check_loan.html")

def loan_prediction(request):
    return render(request, "loan_prediction.html")
 




domain_knowledge = {
    'Account_status': {
        "description": "Status of existing checking account",
        "values": {
            "A11": "< 0 DM",
            "A12": "0 <= ... < 200 DM",
            "A13": ">= 200 DM",
            "A14": "no checking account",
        }
    },

    'Months':{
        "description": "Duration in months",
    },

    'Credit_history':{
        "description": "Credit history",
        "values": {
            "A30": "no credits taken/ all credits paid back duly",
            "A31": "all credits at this bank paid back duly",
            "A32": "existing credits paid back duly till now",
            "A33": "delay in paying off in the past",
            "A34": "critical account/ other credits existing (not at this bank)",
        }
    },

    'Purpose':{
        "description": "Loan Purpose",
        "values": {
            "A40": "car (new)",
            "A41": "car (used)",
            "A42": "furniture/equipment",
            "A43": "radio/television",
            "A44": "domestic appliances",
            "A45": "repairs",
            "A46": "education",
            "A48": "retraining",
            "A49": "business",
            "A410": "others",
        }
    },

    'Credit_amount':{
        "description": "Credit amount",
    },

    'Savings':{
        "description": "Savings account/bonds",
        "values": {
            "A61": "< 100 DM",
            "A62": "100 <= ... < 500 DM",
            "A63": "500 <= ... < 1000 DM",
            "A64": ">= 1000 DM",
            "A65": "unknown/ no savings account",
        }
    },

    'Employment':{
        "description": "Present employment since",
        "values": {
            "A71": "unemployed",
            "A72": "< 1 year",
            "A73": "1 <= ... < 4 years",
            "A74": "4 <= ... < 7 years",
            "A75": ">= 7 years",
        }
    },

    'Installment_rate':{
        "description": "Installment rate in percentage of disposable income",
    },

    'Personal_status':{
        "description":"Personal status and sex",
        "values": {
            	"A91" : "male: divorced/separated",
                "A92" : "female: divorced/separated/married",
                "A93" : "male: single",
                "A94": "male: married/widowed",
                "A95":"female: single",
        }

    },

    "Other_debtors":{
        "description": "Other debtors/ guarantors",
        "values": {
            "A101": "none",
            "A102": "co-applicant",
            "A103": "guarantor",
        }
    },

    "Residence":{
        "description": "Present residence since",
    },

    "Property":{
        "description": "Property type",
        "values": {
            "A121": "real estate",
            "A122": "if not real estate: building society savings agreement/ life insurance",
            "A123": "if not real estate/building society savings agreement/ life insurance:  car or other, not in Savings account/bonds",
            "A124": "unknown / no property",
        }
    },

    "Age":{
        "description": "Age in years",
    },

    "Other_installments":{
        "description": "Other installments",
        "values": {
            "A141": "bank",
            "A142": "stores",
            "A143": "none",
        }
    },

    "Housing":{
        "description": "Housing situation",
        "values": {
            "A151": "rent",
            "A152": "own",
            "A153": "for free",
        }
    },

    "Number_credits":{
        "description": "Number of existing credits at this bank",
    },

    "Job":{
        "description": "Employment status",
        "values": {
            "A171": "unemployed/ unskilled - non-resident",
            "A172": "unskilled - resident",
            "A173": "skilled employee / official",
            "A174": "management/ self-employed/ highly qualified employee/ officer",
        }
    },

    "Number_dependents":{
        "description": "Number of people being liable to provide maintenance for",
    },

    "Telephone":{
        "description": "Telephone registration",
        "values": {
            "A191": "none",
            "A192": "yes",
        }
    },

    "Foreign_worker":{
        "description": "Foreign worker",
        "values": {
            "A201": "yes",
            "A202": "no",
        }
    },

    "prediction": {
        "values": {
            "1": "Creditworthy",
            "0": "Non-Creditworthy",
        }
    }
}
def identify_changes(counterfactual_instance, input_df):
    changes = {}
    for feature in input_df.columns:
        original_value = input_df[feature].iloc[0]
        new_value = counterfactual_instance[feature]
        if original_value != new_value:
            changes[feature] = (original_value, new_value)
    return changes

def create_explanation_graph(counterfactual_instances, prediction, input_df):
    no_cf = len(counterfactual_instances)
    G = nx.DiGraph()
    G.add_node("Explanation", label = "Explanation")
    G.add_node(prediction, label = f"Model Output: {prediction}")
    G.add_edge("Explanation", prediction, label = "Model Prediction")

    pred_meaning = domain_knowledge["prediction"]["values"][str(prediction)]
    G.add_node(pred_meaning, label = pred_meaning)
    G.add_edge(prediction, pred_meaning, label = "Prediction Meaning")

    G.add_node("Current Features", label = "Current Features")
    G.add_edge("Explanation", "Current Features", label = "Current Situation")

    for i in range(no_cf):
        changes = identify_changes(counterfactual_instances.iloc[i], input_df)

        G.add_node(f"Counterfactual{i+1} Features", label = f"Counterfactual{i+1} Features")
        G.add_edge("Explanation", f"Counterfactual{i+1} Features", label = "Alternative Situation")


        for feature, (original_value, new_value) in changes.items():
            G.add_node(f"Current {feature}", label= feature)
            G.add_edge("Current Features", f"Current {feature}", label="Current Feature")

            feature_descr = domain_knowledge[feature]["description"]
            G.add_node(f"Current {feature_descr}", label=feature_descr)
            G.add_edge(f"Current {feature}", f"Current {feature_descr}", label = "Description")


            G.add_node(f"New cf{i+1} {feature}", label=feature)
            G.add_edge(f"Counterfactual{i+1} Features", f"New cf{i+1} {feature}", label="Changed Feature")
            G.add_node(f"New cf{i+1} {feature_descr}", label=feature_descr)
            G.add_edge(f"New cf{i+1} {feature}", f"New cf{i+1} {feature_descr}", label = "Description")

            G.add_node(f"Current {original_value}", label=original_value)
            G.add_edge(f"Current {feature}", f"Current {original_value}", label="Current Value")
            G.add_node(f"New cf{i+1} {new_value}", label=new_value)
            G.add_edge(f"New cf{i+1} {feature}", f"New cf{i+1} {new_value}", label="Changed Value")

           
            if "values" in domain_knowledge[feature]:
                original_val_descr = domain_knowledge[feature]["values"][original_value]
                G.add_node(f"Current {original_val_descr}", label = original_val_descr)
                G.add_edge(f"Current {original_value}", f"Current {original_val_descr}", label="Describes")

                new_val_descr = domain_knowledge[feature]["values"][new_value]
                G.add_node(f"New cf{i+1} {new_val_descr}", label = new_val_descr)
                G.add_edge(f"New cf{i+1} {new_value}", f"New cf{i+1} {new_val_descr}", label="Describes")
    return G


def depth_first_search(graph, node, visited, explanation, idx = 0):
    visited.add(node)
    for neighbor in graph.neighbors(node):
        relation = graph.get_edge_data(node, neighbor)["label"]

        if relation == "Prediction Meaning":
            explanation.insert(0, f"The prediction is '{neighbor}' <br><br>")
        else:
            if relation == "Current Situation":
                explanation.append(f"<br> Current Situation: <br><ul>")
            if relation == "Alternative Situation":
                explanation.append(f"</ul><br> Alternative Situation: <br><ul>")
            if graph.out_degree(neighbor) == 0:
                idx += 1 
                neighbor_label = graph.nodes[neighbor]["label"]
                if idx % 2 == 0:
                    explanation.append(f"{neighbor_label} </li>")
                if idx % 2 != 0:
                    explanation.append(f"<li>{neighbor_label}: ")
        if neighbor not in visited:
            depth_first_search(graph, neighbor, visited, explanation, idx)
    return explanation

def generate_nle(graph, root_node):
    explanation = []
    explanation = depth_first_search(graph, root_node, set(), explanation)
    print(explanation)
    
    return "".join(explanation)

def generate_counterfactual(input_data, prediction):
    
    input_df = pd.DataFrame(input_data, index=[0])
    
    data = dice_ml.Data(features={'Account_status': ['A11', 'A12', 'A13', 'A14'],
                            'Months': [4, 72],
                            'Credit_history': ['A30', 'A31', 'A32', 'A33', 'A34'],
                            'Purpose': ['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A48', 'A49', 'A410'],
                            'Credit_amount': [250, 18424],
                            'Savings': ['A61', 'A62', 'A63', 'A64', 'A65'],
                            'Employment': ['A71', 'A72', 'A73', 'A74', 'A75'],
                            'Installment_rate': [1, 4],
                            'Personal_status': ['A91', 'A92', 'A93', 'A94'],
                            'Other_debtors': ['A101', 'A102', 'A103'],
                            'Residence': [1, 4],
                            'Property': ['A121', 'A122', 'A123', 'A124'],
                            'Age': [19, 75],
                            'Other_installments': ['A141', 'A142', 'A143'],
                            'Housing': ['A151', 'A152', 'A153'],
                            'Number_credits': [1, 4],
                            'Job': ['A171', 'A172', 'A173', 'A174'],
                            'Number_dependents': [1, 2],
                            'Telephone': ['A191', 'A192'],
                            'Foreign_worker': ['A201', 'A202']    
                           },
                 outcome_name='target')
    fixed_features = ["Credit_history", "Personal_status", "Age", "Number_dependents", "Foreign_worker"]
    features_to_vary = list(set(data.feature_names) - set(fixed_features))


    m = dice_ml.Model(model=model1, backend="sklearn", model_type='classifier')
    exp = dice_ml.Dice(data, m, method="random")
    e = exp.generate_counterfactuals(input_df, total_CFs=2, desired_class="opposite", features_to_vary=features_to_vary)
    # exp_user.visualize_as_dataframe(show_only_changes=True)
    e_df = e.cf_examples_list[0].final_cfs_df
    # print(exp_df)
    G = create_explanation_graph(e_df, prediction, input_df)
    nle = generate_nle(G, "Explanation")
    return nle


def generate_feature_importance(input_data):
    explainer = shap.Explainer(model1)
    input_df = pd.DataFrame(input_data, index=[0])
    shap_values = explainer(input_df)
    
    return shap_values



domain_knowledge_shap = {
    'Account_status': {
        "A11": "High risk. Financial instability",
        "A12": "Potential high risk. Financial instability",
        "A13": "Positive. Consider expenses and debt factors",
        "A14": "High risk. No financial history",
    },
    'Months':{
        "<36 months": "Higher monthly payments. Less risk for lender. Consider income, expenses and existing debt.",
        "36-72 months": "Higher risk due to uncertainty in future income",
    },
    'Credit_history':{
        "A30": "Consider building credit history at this bank",
        "A31": "Positive factor. Consider other factors, such as not borrowing excessively",
        "A32": "Consider factors such as over-borrowing, debt and income",
        "A33": "Increased risk of default",
        "A34": "Consider paying off existing loans before taking new ones",
    },
    'Purpose':{
        "A40": "Considerable financial comittement. Ensure good debt to income ratio",
        "A41": "While more affordable than a new car, ensure good debt to income ratio",
        "A42": "Non-essential, lower priority. Quickly deapreciates, which means the lender may not recover loan through resale",
        "A43": "Non-essential, lower priority. Quickly deapreciates, which means the lender may not recover loan through resale",
        "A44": "Non-essential, lower priority. Quickly deapreciates, which means the lender may not recover loan through resale",
        "A45": "Needing repairs might indicate a lack of savings or financial instability",
        "A46": "Debt to income ratio concerns. Uncertainty about return on investment.",
        "A48": "Uncertainty. More risky. Look for affordable retraining options. Ensure financial stabilty",
        "A49": "High risk investment. ",
        "A410": "Lack of detailed information",
    },
    'Credit_amount':{
        "250-6000": "Low amount, low risk. Generally positive, but other factors might be unfavorable",
        "6000-12000": "Moderate amount, moderate risk. Consider other factors or lower credit amount",
        "12000-18424": "High amount, high risk. Consider other factors or lower credit amount",
    },
    'Savings':{
        "A61": "Low savings. Higher risk. Consider building savings before taking loans to show financial stability",
        "A62": "Might be insufficient for required loan amount",
        "A63": "Moderate savings. Consider building savings before taking loans",
        "A64": "Good savings are generally positive. Consider other factors",
        "A65": "Absence of savings might indicate financial instability and higher risk for lender",
    },
    'Employment':{
        "A71": "No income source is high risk.",
        "A72": "Short employment history is a risk factor due to uncertainty in income",
        "A73": "Stable employment. Generally positive, but consider other factors or the requied loan amount requires better stability",
        "A74": "High level of stable employment. Generally positive, but consider other factors",
        "A75": "Stable employment. Generally positive, but consider other factors, such as savings and creditworthiness",
    },
    'Installment_rate':{
        "explanation": "High installment rate means a large proportion of income is dedicated to lean repayment. This can be risky."
    },
    'Personal_status':{
        'explanation': 'Family responsibilities can increase financial comittments and reduce flexibility, '
        'consider income levels, savings and debt. Single individuals have fewer obligations, but depending on income, debt, '
        'employment history, the lack of financial support might be risky',
    },
    'Other_debtors':{
        "A101": "No other debtors means no shared responsibility, which means that it reduces changes of loan acceptance, especially high risk applicants",
        "A102": "Reduces risks for lender, but consider co-applicant financial profile and other factors",
        "A103": "Reduces risks for lender, but consider guarantor financial profile and other factors",
    },
    'Residence':{
        "explanation": "Short periods of residence increase risks, as they may indicate financial or employment instability."
    },
    'Property':{
        "A121": "Generally a positive factor, but consider other factors, such as debt for the property",
        "A122": "Generally a positive factor, but consider other factors, such as savings",
        "A123": "Cars deapreciate quickly. Consider other factors, such as investing in assets that increase in value over time or increase savings",
        "A124": "Absence of assets increases the risk level for the lender, as they can not be used as collaterals in case of non payment",
    },
    'Age':{
        "19-25": "Younger applicants may have less financial stability and less credit history. Low income, student debts",
        "26-35": "Young adults may have some financial stability, but there is a risk of financial stress due to multiple debts, such as student loans"
        "and mortgages",
        "36-45": "Middle-aged applicants may have more financial stability, but there is a risk of financial stress due to multiple debts, such as mortgages"
        "and other loans",
        "46-55": "Older applicants may have more financial stability, but there is a risk of financial stress due to multiple debts and insufficient savings",
        "56-65": "Older applicants near retirement have limited future earning potential, which can increase credit risk without sufficient savings",

    },
    'Other_installments':{
        "A141": "Depending on the installment type and income, might increase the risk of inability to pay the loan",
        "A142": "While generally less risky than bank loans, it can still increase the risk of inability to pay the loan. Multiple loans "
        "might lead to high debt to income ratio",
        "A143": "Generally positive, but without a credit history, few assets, or low savings, it is difficult to assess if applicant can repay loan",
    },
    'Housing':{
        "A151": "Renting means no assets to use as collateral and monthly payment comittements. Consider rent amount and other factors such as income and debt",
        "A152": "Owning a home is generally a positive factor, but consider other factors, such as debt for the property, property expenses and income",
        "A153": "Might indicate financial instability. Consider other factors.",
    },
    'Number_credits':{
        "explanation": "Multple credits might indicate financial instability, higher debt to income ratio, higher debt levels and high risk for lender.",
    },
    'Job':{
        "A171": "Unemployed or unskilled workers are high risk for lenders due to lower income levels or financial instability. Non residency "
        "limits access to job opportunities",
        "A172": "This can indicate limited earning potential. Temporary jobs indicate instability and uncertainty in income. ",
        "A173": "Generally positive, but consider other factors, such as income and debt levels",
        "A174": "Generally positive. Self-employment can be volatile. Consider income, debt levels.",
    },
    'Number_dependents':{
        "explanation": "Higher number of dependents means higher financial comittments and less flexibility. "
        "Consider other factos such as income, debt and savings",
    },
    'Telephone':{
        "A191": "Might indicate financial instability, and higher risks for lender",
        "A192": "Telephone registration is generally a positive factor, but consider other factors",
    },
    'Foreign_worker':{
        "A201": "May not have enough credit history.",
        "A202": "Consider other factors",
    },

}

protected_attributes = {
    "Personal_status": {
        "protected_attr": "sex",
    },
    "Age": {
        "protected_attr": "age",
    },
    "Foreign_workers": {
        "protected_attr": "race, ethnicity and social origin",
    },
}


def shap_graph(features, data, feature_contributions, prediction):
    G = nx.DiGraph()
    G.add_node("Explanation", label ="Explanation")
    G.add_node(prediction, label = f"Model Output: {prediction}")
    G.add_edge("Explanation", prediction, label = "Model Prediction")

    pred_meaning = domain_knowledge["prediction"]["values"][str(prediction)]
    G.add_node(pred_meaning, label = pred_meaning)
    G.add_edge(prediction, pred_meaning, label = "Prediction Meaning")

    G.add_node("Negative Contribution", label = "Negative Contribution")
    G.add_edge("Explanation", "Negative Contribution", label = "Negative Contributions")

    G.add_node("Protected attributes", label = "Protected attributes")
    G.add_edge("Explanation", "Protected attributes", label = "Protected attributes")

    


    for i in range (len(features)):
        G.add_node(features[i], label = features[i])
        G.add_edge("Negative Contribution", features[i], label = f"Contributes {feature_contributions[i]}")
        feature_descr = domain_knowledge[features[i]]["description"]
        G.add_node(feature_descr, label = feature_descr)
        G.add_edge(features[i], feature_descr, label = "Describes")
        
        G.add_node(data[i], label = data[i])
        G.add_edge(features[i], data[i], label = "Value of feature")

        

            
        if "values" in domain_knowledge[features[i]]:
            data_val = domain_knowledge[features[i]]["values"][data[i]]
            G.add_node(data_val, label = data_val)
            G.add_edge(data[i], data_val, label = "Describes")

            if features[i] != "Personal_status":
                data_exp = domain_knowledge_shap[features[i]][data[i]]
                G.add_node(data_exp, label = f"{data[i]} explanation")
                G.add_edge(data[i], data_exp, label = "Value Explanation")

        if features[i] == "Months":
            if data[i] < 36:
                exp = domain_knowledge_shap[features[i]['<36 months']]
                G.add_node(exp, label = "short loan")
                G.add_edge(data[i], exp, label = "Value Explanation")
            else:
                exp = domain_knowledge_shap[features[i]['36-72 months']]
                G.add_node(exp, label = "long loan")
                G.add_edge(data[i], exp, label = "Value Explanation")
        
        if features[i] == "Credit_amount":
            if data[i] < 6000:
                exp = domain_knowledge_shap[features[i]['250-6000']]
                G.add_node(exp, label = "low amount")
                G.add_edge(data[i], exp, label = "Value Explanation")
            elif data[i] < 12000:
                exp = domain_knowledge_shap[features[i]['6000-12000']]
                G.add_node(exp, label = "moderate amount")
                G.add_edge(data[i], exp, label = "Value Explanation")
            else:
                exp = domain_knowledge_shap[features[i]['12000-18424']]
                G.add_node(exp, label = "high amount")
                G.add_edge(data[i], exp, label = "Value Explanation")
        
        if features[i] == "Installment_rate":
            exp = domain_knowledge_shap[features[i]]["explanation"]
            G.add_node(exp, label = "installment rate")
            G.add_edge(data[i], exp, label = "Value Explanation")
        
        if features[i] == "Personal_status":
            exp = domain_knowledge_shap[features[i]]["explanation"]
            G.add_node(exp, label = "personal status")
            G.add_edge(data[i], exp, label = "Value Explanation")
        
        if features[i] == "Residence":
            exp = domain_knowledge_shap[features[i]]["explanation"]
            G.add_node(exp, label = "residence")
            G.add_edge(data[i], exp, label = "Value Explanation")
        
        if features[i] == "Age":
            if data[i] < 25:
                exp = domain_knowledge_shap[features[i]['19-25']]
                G.add_node(exp, label = "young")
                G.add_edge(data[i], exp, label = "Value Explanation")
            elif data[i] < 35:
                exp = domain_knowledge_shap[features[i]['26-35']]
                G.add_node(exp, label = "young adult")
                G.add_edge(data[i], exp, label = "Value Explanation")
            elif data[i] < 45:
                exp = domain_knowledge_shap[features[i]['36-45']]
                G.add_node(exp, label = "middle aged")
                G.add_edge(data[i], exp, label = "Value Explanation")
            elif data[i] < 55:
                exp = domain_knowledge_shap[features[i]['46-55']]
                G.add_node(exp, label = "older adult")
                G.add_edge(data[i], exp, label = "Value Explanation")
            else:
                exp = domain_knowledge_shap[features[i]['56-65']]
                G.add_node(exp, label = "senior")
                G.add_edge(data[i], exp, label = "Value Explanation")
        
        if features[i] == "Number_credits":
            exp = domain_knowledge_shap[features[i]]
            G.add_node(exp, label = "number of credits")
            G.add_edge(data[i], exp, label = "Value Explanation")
        
        if features[i] == "Number_dependents":
            exp = domain_knowledge_shap[features[i]]
            G.add_node(exp, label = "number of dependents")
            G.add_edge(data[i], exp, label = "Value Explanation")
        
        if features[i] in protected_attributes:
            protected_attr = protected_attributes[features[i]]["protected_attr"]
            G.add_node(protected_attr, label = protected_attr )
            G.add_edge("Protected attributes", protected_attr, label = "Protected attribute")
    return G


def depth_first_search_shap(graph, node, visited, exp_pred, exp_outcome, exp_protected, idx = 0):
    visited.add(node)
    for neighbor in graph.neighbors(node):
        relation = graph.get_edge_data(node, neighbor)["label"]
        if relation == "Prediction Meaning":
            ## explanation.insert(0, f"<br> The prediction of '{neighbor}' is based on the following top 5 factors:<ul>")
            exp_pred.append(f"<br> The prediction of '{neighbor}' is based on the following top 5 factors:<ul>")

        if graph.out_degree(neighbor) == 0 and relation != "Protected attribute" and relation != "Protected attributes" and relation != "Prediction Meaning":
            idx += 1
            if idx % 3 == 0:
                exp_outcome.append(f"{neighbor}: </li>")
            elif idx % 3 == 1:
                exp_outcome.append(f"<li>{neighbor}: ")
            else:
                label = graph.nodes[neighbor]["label"]
                exp_outcome.append(f" {label}  </li>")

        if relation == "Protected attributes":
            exp_protected.append(f"<br> Protected attributes, according to the EU Charter of Fundamental Rights: <br><ul>")
        if relation == "Protected attribute":
            exp_protected.append(f"<li> {neighbor} </li>")
       
        if neighbor not in visited:
            depth_first_search_shap(graph, neighbor, visited, exp_pred, exp_outcome, exp_protected, idx)
    return exp_pred, exp_outcome, exp_protected


def shap_nle(graph, root_node):
    explanation = []
    exp_pred, exp_outcome, exp_protected = [], [], []
    exp_pred, exp_outcome, exp_protected = depth_first_search_shap(graph, root_node, set(), exp_pred, exp_outcome, exp_protected)
    exp_outcome.append("</ul>")
    exp_protected.append("</ul>")
    if len(exp_protected) > 0:
        exp_protected.append("If you believe that protected attributes contributed to a dicriminatory negative prediction, please contact support or a human reviwer.")
    
    explanation = exp_pred + exp_outcome + exp_protected
    
    return "".join(explanation)


def chatbot(request):
    if request.method == "POST":
        prediction = request.session.get("prediction")
        loan_data = request.session.get("loan_data")
        print(prediction)
        

        if not prediction or not loan_data:
            return JsonResponse({'error': 'Please submit a loan application form.'})
        

        user_input = request.POST.get("user_input")


        if user_input == "Remind me my loan status?":
            response = ""
            if prediction[0] == "1":
                response = "Creditworthy"
            else:
                response = "Non-Creditworthy"
            return JsonResponse({'answer': response})
        
        if user_input == "How would my loan status be different?":
            response = generate_counterfactual(loan_data, prediction)
            return JsonResponse({'answer': response})
        
        if user_input == "What features contributed to the prediction?":
            shap_values = generate_feature_importance(loan_data)
            feature_contributions = shap_values.values[0, :]
            feature_names = shap_values.feature_names
            data = shap_values.data[0, :]
            print(data)
            
            
            ### Show top 5 features with negative contributions
            sorted_indices = np.argsort(feature_contributions)
            top_indices = sorted_indices[:5]
            feature_contributions = feature_contributions[top_indices]
            feature_names = [feature_names[i] for i in top_indices]
            data = data[top_indices]

            G_shap = shap_graph(feature_names, data, feature_contributions, prediction)
            pos = nx.spring_layout(G_shap) 
            
            edge_labels = nx.get_edge_attributes(G_shap, 'label')
            
            nx.draw(G_shap, pos, with_labels=True, node_size = 900, font_size = 10)
            nx.draw_networkx_edge_labels(G_shap, pos, edge_labels=edge_labels)
            #plt.savefig("shap_graph_nx.png")
            #plt.close()

            A = nx.nx_agraph.to_agraph(G_shap)
            A.layout(prog='dot')
            A.draw('shap_graph.pdf')

            response = shap_nle(G_shap, "Explanation")
            


            # response = "The following features contributed to the prediction: "
            return JsonResponse({'answer': response})

        return JsonResponse({'answer': "Sorry, I didn't understand that."})
    return render(request, "chatbot.html")


def other_chatbot(request):
    if request.method == "POST":
        user_input = request.POST.get("user_input")
        
        if user_input == "I want to see which features are most important globally":
            X_PATH = os.path.join(os.path.dirname(__file__), "statlog_german_credit_data")

            german_credit_data = pd.read_csv(X_PATH, index_col=0) 
            german_credit_data.columns = ['Account_status', 'Months', 'Credit_history', 'Purpose', 
                              'Credit_amount', 'Savings', 'Employment', 'Installment_rate', 
                              'Personal_status', 'Other_debtors', 'Residence', 'Property', 'Age', 
                              'Other_installments', 'Housing', 'Number_credits', 'Job', 'Number_dependents', 
                              'Telephone', 'Foreign_worker', 'target'] 
            X = german_credit_data.loc[:, german_credit_data.columns != "target"]
            explainer = shap.Explainer(model1)
            
            shap_values = explainer(X)
            #feature_contributions = shap_values.values
            #feature_names = shap_values.feature_names
            #data = shap_values.data
            #print(feature_contributions.shape)
            #print(len(feature_names))
            #print(data.shape)
            #print(X.shape)
            # change shap values to X_test
            mean_abs_values = np.abs(shap_values.values).mean(axis=0)

            feature_importance = pd.DataFrame({
                'Feature': shap_values.feature_names,
                'Mean Absolute Value': mean_abs_values,
            })

            feature_importance = feature_importance.sort_values(by='Mean Absolute Value', ascending=False)
            print(feature_importance.head(4))
            response = "<br>"
            for index, row in feature_importance.iterrows():
                print(f"Feature: {row['Feature']}, Mean Absolute Value: {row['Mean Absolute Value']}")
                response += "Feature: " + row['Feature'] + ", Mean Absolute Value: " + str(row['Mean Absolute Value']) + "<br>"
                
            print(response)

            # shap.summary_plot(shap_values, X)
            return JsonResponse({'answer': response})
        if user_input == "I want to see the fairness of the model":
            X_PATH = os.path.join(os.path.dirname(__file__), "statlog_german_credit_data")

            german_credit_data = pd.read_csv(X_PATH, index_col=0) 
            german_credit_data.columns = ['Account_status', 'Months', 'Credit_history', 'Purpose', 
                              'Credit_amount', 'Savings', 'Employment', 'Installment_rate', 
                              'Personal_status', 'Other_debtors', 'Residence', 'Property', 'Age', 
                              'Other_installments', 'Housing', 'Number_credits', 'Job', 'Number_dependents', 
                              'Telephone', 'Foreign_worker', 'target'] 
            X = german_credit_data.loc[:, german_credit_data.columns != "target"]
            y = german_credit_data["target"]
            y = y.replace({1: 0, 2: 1})
            X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42) # 80% training and 20% test

            

            y_pred = model1.predict(X_test)
            #accuracy = np.mean(y_pred == y_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            #print(f"Precision: {precision}")
            #print(f"Accuracy: {accuracy}")
            

            personal_status = X_test["Personal_status"]
            age = X_test["Age"]
            foreign_worker = X_test["Foreign_worker"]
            
            metric_frame_personal_status = MetricFrame(
                metrics={"accuracy": accuracy_score, 
                         # "precision": precision_score, 
                         # "recall": recall_score, 
                         # "f1": f1_score,
                         # "selection_rate": selection_rate, 
                         # "count": count, 
                         # "false_positive_rate": false_positive_rate, 
                         # "false_negative_rate": false_negative_rate
                         },
                y_true=y_test,
                y_pred=y_pred,
                sensitive_features=personal_status,
            )
            # print("demographic parity")
            # print(demographic_parity_ratio(y_test, y_pred, sensitive_features=personal_status))
            # print("equalized odds")
            # print(equalized_odds_ratio(y_test, y_pred, sensitive_features=personal_status))
            #print(metric_frame_personal_status.overall)
            #print(metric_frame_personal_status.by_group)



            metric_frame_age = MetricFrame(
                metrics={"accuracy": accuracy_score, 
                         # "precision": precision_score, 
                         # "recall": recall_score, 
                         # "f1": f1_score,
                         # "selection_rate": selection_rate, 
                         # "count": count, 
                         # "false_positive_rate": false_positive_rate, 
                         # "false_negative_rate": false_negative_rate
                         },
                y_true=y_test,
                y_pred=y_pred,
                sensitive_features=age,
            )
            
            # print(metric_frame_age.overall)
            # print(metric_frame_age.by_group)

            metric_frame_foreign_worker = MetricFrame(
                metrics={"accuracy": accuracy_score, 
                         # "precision": precision_score, 
                         # "recall": recall_score, 
                         # "f1": f1_score,
                         # "selection_rate": selection_rate, 
                         # "count": count, 
                         # "false_positive_rate": false_positive_rate, 
                         # "false_negative_rate": false_negative_rate
                         },
                y_true=y_test,
                y_pred=y_pred,
                sensitive_features=foreign_worker,
            )
            # print(metric_frame_foreign_worker.overall)
            # print(metric_frame_foreign_worker.by_group)

            feature_name = "Foreign_worker"
            feature_index = list(X_test.columns).index(feature_name)
            # print(feature_index)

            X = german_credit_data.loc[:, german_credit_data.columns != "target"]
            explainer = shap.Explainer(model1)
            
            shap_values = explainer(X)

            abs_shap_values = np.abs(shap_values.values)
            df_shap = pd.DataFrame(abs_shap_values, columns=shap_values.feature_names)
            df_shap["Group"] = X["Foreign_worker"]
            # print(df_shap.head(3))

            shap_importance_foreign_worker = df_shap.groupby("Group").mean()
            # print(shap_importance_foreign_worker.head(3))
            # shap_importance_foreign_worker= shap_importance_foreign_worker.T
            # print(shap_importance_foreign_worker)

            group_1 = shap_importance_foreign_worker.iloc[0]

            group_1 = group_1.sort_values(ascending=False)
            # print(group_1.head(3))
            group_2 = shap_importance_foreign_worker.iloc[1]
            group_2 = group_2.sort_values(ascending=False)
            # print(group_2.head(3))
            # shap.dependence_plot("Foreign_worker", shap_values, X, shap_values.feature_names)
            categorical_features = ['Account_status', 'Credit_history', 'Purpose', 'Savings', 'Employment', 'Personal_status', 'Other_debtors', 'Property', 'Other_installments', 'Housing', 'Job', 'Telephone', 'Foreign_worker']
            data_pool = Pool(X, y, cat_features=categorical_features)


            interactions = model1.get_feature_importance(data_pool, type = "Interaction")
            print(interactions.shape)

            interactions_df = pd.DataFrame(interactions, columns=["Feature 1 Index", "Feature 2 Index", "Interaction Strength"])
            feature_names = X.columns
            interactions_df["Feature 1"] = interactions_df["Feature 1 Index"].apply(lambda x: feature_names[int(x)])
            interactions_df["Feature 2"] = interactions_df["Feature 2 Index"].apply(lambda x: feature_names[int(x)])
            interactions_df = interactions_df.sort_values(by="Interaction Strength", ascending=False)
            #print(interactions_df[["Feature 1", "Feature 2", "Interaction Strength"]].head(10))
            specific_feature = "Credit_history"
            specific_feature_interactions = interactions_df[
                 (interactions_df["Feature 1"] == specific_feature) | (interactions_df["Feature 2"] == specific_feature)    
            ]   

            # Sort by interaction strength
            specific_feature_interactions = specific_feature_interactions.sort_values(by="Interaction Strength", ascending=False)

            # Display the interactions
            print(specific_feature_interactions[["Feature 1", "Feature 2", "Interaction Strength"]])
            

            feature_importances = model1.get_feature_importance()
            print(pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False))
            print(X["Foreign_worker"].value_counts())
            return JsonResponse({'answer': "Sorry, I didn't understand that."})


        return JsonResponse({'answer': "Sorry, I didn't understand that."})
    return render(request, "other_chatbot.html")