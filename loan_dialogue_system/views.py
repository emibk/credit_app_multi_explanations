from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required, user_passes_test
import os
import pickle
import pandas as pd
import dice_ml
from dice_ml.utils import helpers
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fairlearn.metrics import demographic_parity_ratio, demographic_parity_difference
from fairlearn.metrics import equalized_odds_ratio, equalized_odds_difference
from fairlearn.metrics import equal_opportunity_difference, equal_opportunity_ratio
import pygraphviz as pgv
import shap
from fairlearn.metrics import MetricFrame, selection_rate, count, false_positive_rate, false_negative_rate, true_positive_rate, true_negative_rate, mean_prediction
from catboost import Pool
# Create your views here.

def index(request):
    return render(request, "index.html")

MODEL_PATH1 = os.path.join(os.path.dirname(__file__), "catboost_model_1.pkl")

with open(MODEL_PATH1, "rb") as f:
    model1 = pickle.load(f)

def is_loan_applicant(user):
    return user.groups.filter(name='loan_applicant').exists()

@login_required
@user_passes_test(is_loan_applicant)
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
          
            input_df = pd.DataFrame(input_data, index=[0])

            prediction = model1.predict(input_df)
           
            
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

    'prediction': {
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
    G = pgv.AGraph(directed=True)
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
                G.add_edge(f"Current {original_value}", f"Current {original_val_descr}", label="Description")

                new_val_descr = domain_knowledge[feature]["values"][new_value]
                G.add_node(f"New cf{i+1} {new_val_descr}", label = new_val_descr)
                G.add_edge(f"New cf{i+1} {new_value}", f"New cf{i+1} {new_val_descr}", label="Description")
    return G


def depth_first_search(graph, node, visited, explanation, idx = 0):
    visited.add(node)
    print(node)
    print(graph.neighbors(node))
    for neighbor in graph.neighbors(node):
        if graph.has_edge(node, neighbor):
            print(f"Edge from {node} to {neighbor}")
        else: 
            continue
        relation = graph.get_edge(node, neighbor).attr['label']

        if relation == "Prediction Meaning":
            explanation.insert(0, f"The prediction is '{neighbor}' <br><br>")
        else:
            if relation == "Current Situation":
                explanation.append(f"<br> Current Situation: <br><ul>")
            if relation == "Alternative Situation":
                explanation.append(f"</ul><br> Alternative Situation: <br><ul>")
            if len(graph.out_edges(neighbor)) == 0:
                idx += 1 
                neighbor_label = graph.get_node(neighbor).attr['label']
                
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
    e_df = e.cf_examples_list[0].final_cfs_df
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
        "A11": "Negative balance suggests financial instability and difficulties repaying the loan. High risk for the lender.",
        "A12": "A small positive balance indicates potential difficulties with unexpected expenses, and may be insufficient in demonstrating the ability to repay the loan.",
        "A13": "Typically a positive factor. Consider other factors, such as expenses and debt.",
        "A14": "Lack of financial history that can be assessed by the lender. Consider opening a checking account.",
    },
    'Months':{
        "<36 months": "Shorter loans are less risky for lenders, but require higher monthly payments. Consider factors such as income, expenses and existing debt, which may indicate potential difficulties in repaying the loan.",
        "36-72 months": "Long-term loans pose more risks for lenders, due to uncertainties about future income.",
    },
    'Credit_history':{
        "A30": "Loans can be denied due to a lack of verifiable credit history. Consider building credit history at this bank.",
        "A31": "Typically a positive factor. Consider other factors, such as income, debt and borrowing history.",
        "A32": "Typically a positive factor. Consider other factors, such as income, debt and borrowing history.",
        "A33": "Might indicate financial difficulties. Potential risk of inability to repay the loan.",
        "A34": "Might indicate financial difficulties and high risk of inability to repay the loan. Consider paying off existing loans before taking new ones.",
    },
    'Purpose':{
        "A40": "Involves a considerable financial commitment. Consider factors such as income, debt and expenses.",
        "A41": "While more affordable than a new car, consider factors such as income, maintenance and repair costs, and current expenses.",
        "A42": "Non-essential, lower priority. Quickly deapreciates, which means the lender may not recover loan through resale if the loan is not repaid.",
        "A43": "Non-essential, lower priority. Quickly deapreciates, which means the lender may not recover loan through resale if the loan is not repaid..",
        "A44": "Non-essential, lower priority. Quickly deapreciates, which means the lender may not recover loan through resale if the loan is not repaid..",
        "A45": "Loan can be denied if the borrower's financial stuation is unstable.",
        "A46": "Loan can be denied if the borrower's financial stuation is unstable and due to uncertaintities regarding the return on investment.",
        "A47": "Non-essential, lower priority. High-risk, as it does not contribute to long-term financial stability.",
        "A48": "Uncertaintities regarding the return on investment. Consider affordable retraining options and ensuring long-term financial stability.",
        "A49": "Expenses for business purposes are risky due to uncertain returns on investments.",
        "A410": "Lack of detailed information.",
    },
    'Credit_amount':{
        "250-6000": "Low amount indicates lower risk for lender. Generally positive, but other factors might be unfavorable.",
        "6000-12000": "Moderate amount indicates moderate risk for lender. Consider other factors or lower credit amount.",
        "12000-18424": "High amount indicates higher risk for lender. Consider other factors or lower credit amount",
    },
    'Savings':{
        "A61": "Low savings amount indicates limited financial stability and higher risks for the lender.Low savings. Consider building savings before taking loans.",
        "A62": "Might be insufficient for required loan, due to insufficient financial stability and uncertaintities regarding future expenses.",
        "A63": "Might be insufficient for required loan, due to insufficient financial stability and uncertaintities regarding future expenses.",
        "A64": "Typically a positive factor. Consider other factors, such as income and debt.",
        "A65": "Absence of savings information might indicate financial instability and poor financial planning, increasing risks for the lender."
    },
    'Employment':{
        "A71": "No income source increases the risk of inability to repay the loan.",
        "A72": "Short employment history is a risk factor due to uncertainty in future income, long-term comittements and long-term financial stability.",
        "A73": "Typically a positive factor. Consider other factors and if the requied loan amount requires better financial stability.",
        "A74": "Typically a positive factor. Consider factors such as income and debt.",
        "A75": "Typically a positive factor. Consider factors such as income and debt.",
    },
    'Installment_rate':{
        "explanation": "Higher installment rate means a larger proportion of income is dedicated to lean repayment. Assess the ability to repay the loan."
    },
    'Personal_status':{
        'explanation': 'Family responsibilities can increase financial commitments and reduce flexibility in expense management, '
        'consider income levels, savings and debt. Single individuals have fewer obligations, but depending on factors such as income, debt and '
        'employment history, the lack of financial support might increase risks for lenders.',
    },
    'Personal_status':{
        "A91": "Family responsibilities may increase financial commitments. Consider income levels, savings and debt.",
        "A92": "Family responsibilities can increase financial commitments. Consider income levels, savings and debt.",
        "A93": "May indicate lack of financial support. Consider income, savings, debt and other debtors / guarantors",
        "A94": "Family responsibilities can increase financial commitments. Consider income levels, savings and debt.",
        "A95": "May indicate lack of financial support. Consider income, savings, debt and other debtors / guarantors",
    },
    'Other_debtors':{
        "A101": "No other debtors means no shared responsibility in repaying the loan. Might reduce chances of loan acceptance, especially for high risk applicants.",
        "A102": "Reduces risks for lender, but consider co-applicant financial profile and other factors.",
        "A103": "Reduces risks for lender, but consider guarantor financial profile and other factors.",
    },
    'Residence':{
        "explanation": "Short periods of residence may indicate financial or employment instability. Other factors, such as employment and income can also be assessed to determine financial stability."
    },
    'Property':{
        "A121": "Generally a positive factor. Consider other factors, such as income and debt or expenses for the property.",
        "A122": "Generally a positive factor. Consider other factors, such as savings and financial stability.",
        "A123": "Cars deapreciate quickly, therefore these assets might be insufficient.  Consider other factors, such as investing in assets that increase in value over time or increasing savings amount.",
        "A124": "Absence of assets increases the risk level for the lender, since they can not be used as collaterals in case of non payment.",
    },
    'Age':{
        "19-25": "Younger applicants may face reduced financial stability, limited credit history, low income, student debt and limited employment opportunities.",
        "26-35": "Young adults may have some financial stability, but there is a risk of financial stress due to multiple debts, such as student loans"
        "and mortgages.",
        "36-45": "Middle-aged applicants may have more financial stability, but there is a risk of financial stress due to multiple debts, such as mortgages"
        "and other loans.",
        "46-55": "Older applicants may have more financial stability, but there is a risk of financial stress due to multiple debts and insufficient savings.",
        "56-65": "Older applicants near retirement have limited future earning potential, which can increase risks for lender, especially without sufficient savings.",
        "65+": "Older applicants may have limited future earning potential, which can increase risks for lenders, especially without sufficient savings",

    },
    'Other_installments':{
        "A141": "Indicates multiple financial commitments. Depending on installment types and income, it might increase the risk of inability to pay the loan.",
        "A142": "While generally less risky than bank loans, it can still increase the risk of inability to pay the loan. Multiple loans "
        "might lead to high debt to income ratio.",
        "A143": "Generally positive, but without a credit history, few assets, or low savings, it is difficult to assess if applicant can repay loan.",
    },
    'Housing':{
        "A151": "Renting indicates monthly expenses and potential lack of property to used as collateral in case the loan cannot be paid. Consider rent amount and other factors such as income and debt.",
        "A152": "Owning a home is generally a positive factor. Consider other factors, such as debt for the property, property expenses and income.",
        "A153": "Can indicate financial instability.",
    },
    'Number_credits':{
        "explanation": "Multple credits might indicate financial instability and higher debt levels, leading to higher risks for lender.",
    },
    'Job':{
        "A171": "Unemployed or unskilled workers are considered high risk for lenders due to lower income levels or financial instability. Non residency "
        "limits access to job opportunities.",
        "A172": "Might indicate limited earning potential. Temporary jobs indicate instability and uncertainty in income.",
        "A173": "Generally positive, but consider other factors, such as income and debt levels.",
        "A174": "Generally positive. Self-employment can be volatile. Consider income, debt levels.",
    },
    'Number_dependents':{
        "explanation": "Higher number of dependents means higher financial comittments and less flexibility in expense management. "
        "Consider factos such as income, debt and savings.",
    },
    'Telephone':{
        "A191": "Might indicate financial instability and inability to establish consistent communication.",
        "A192": "Telephone registration is generally a positive factor. Consider factors such as income and debt.",
    },
    'Foreign_worker':{
        "A201": "May not have enough credit history. Could indicate limited employment opportunities.",
        "A202": "Consider other factors, such as income and debt.",
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
        "protected_attr": "race or ethnicity",
    },
}

def shap_graph(features, data, feature_contributions, prediction):
    G = pgv.AGraph(directed=True)
    G.add_node("Explanation", label ="Explanation")
    G.add_node(prediction, label = f"Model Output: {prediction}")
    G.add_edge("Explanation", prediction, label = "Model Prediction")

    pred_meaning = domain_knowledge["prediction"]["values"][str(prediction)]
    G.add_node(pred_meaning, label = pred_meaning)
    G.add_edge(prediction, pred_meaning, label = "Prediction Meaning")

    G.add_node("Negative Contributions", label = "Negative Contributions")
    G.add_edge("Explanation", "Negative Contributions", label = "Negative Contributions")

    G.add_node("Protected attributes", label = "Protected attributes")
    G.add_edge("Explanation", "Protected attributes", label = "Protected attributes")

    


    for i in range (len(features)):
        G.add_node(features[i], label = features[i])
        G.add_edge("Negative Contributions", features[i], label = f"Contributes {feature_contributions[i]:.2f}")
        feature_descr = domain_knowledge[features[i]]["description"]
        G.add_node(feature_descr, label = feature_descr)
        G.add_edge(features[i], feature_descr, label = "Description")
        
        G.add_node(data[i], label = data[i])
        G.add_edge(features[i], data[i], label = "Value of feature")

        

            
        if "values" in domain_knowledge[features[i]]:
            data_val = domain_knowledge[features[i]]["values"][data[i]]
            G.add_node(data_val, label = data_val)
            G.add_edge(data[i], data_val, label = "Description")

            
            data_exp = domain_knowledge_shap[features[i]][data[i]]
            G.add_node(data_exp, label = f"{data[i]} explanation")
            G.add_edge(data[i], data_exp, label = "Value Explanation")

        if features[i] == "Months":
            if data[i] < 36:
                G.add_node('<36 months', label = '<36 months')
                G.add_edge(data[i], '<36 months', label = "Description")
                exp = domain_knowledge_shap[features[i]]['<36 months']
                G.add_node(exp, label = "short loan")
                G.add_edge(data[i], exp, label = "Value Explanation")
            else:
                G.add_node('36-72 months', label = '36-72 months')
                G.add_edge(data[i], '36-72 months', label = "Description")
                exp = domain_knowledge_shap[features[i]]['36-72 months']
                G.add_node(exp, label = "long loan")
                G.add_edge(data[i], exp, label = "Value Explanation")
        
        if features[i] == "Credit_amount":
            if data[i] < 6000:
                G.add_node('250-6000', label = '250-6000')
                G.add_edge(data[i], '250-6000', label = "Description")
                exp = domain_knowledge_shap[features[i]]['250-6000']
                G.add_node(exp, label = "low amount")
                G.add_edge(data[i], exp, label = "Value Explanation")
            elif data[i] < 12000:
                G.add_node('6000-12000', label = '6000-12000')
                G.add_edge(data[i], '6000-12000', label = "Description")
                exp = domain_knowledge_shap[features[i]]['6000-12000']
                G.add_node(exp, label = "moderate amount")
                G.add_edge(data[i], exp, label = "Value Explanation")
            else:
                G.add_node('12000-18424', label = '12000-18424')
                G.add_edge(data[i], '12000-18424', label = "Description")
                exp = domain_knowledge_shap[features[i]]['12000-18424']
                G.add_node(exp, label = "high amount")
                G.add_edge(data[i], exp, label = "Value Explanation")
        
        if features[i] == "Installment_rate":
            if data[i] < 2:
                G.add_node('<2%', label = '<2%')
                G.add_edge(data[i], '<2%', label = "Description")
            else:
                G.add_node('2-4%', label = '2-4%')
                G.add_edge(data[i], '2-4%', label = "Description")
            exp = domain_knowledge_shap[features[i]]["explanation"]
            G.add_node(exp, label = "installment rate")
            G.add_edge(data[i], exp, label = "Value Explanation")
        
        
        
        if features[i] == "Residence":
            if data[i] < 2:
                G.add_node('<2 years', label = '<2 years')
                G.add_edge(data[i], '<2 years', label = "Description")
            else:
                G.add_node('2-4 years', label = '2-4 years')
                G.add_edge(data[i], '2-4 years', label = "Description")
            exp = domain_knowledge_shap[features[i]]["explanation"]
            G.add_node(exp, label = "residence")
            G.add_edge(data[i], exp, label = "Value Explanation")
        
        if features[i] == "Age":
            if data[i] <= 25:
                G.add_node('19-25', label = '19-25')
                G.add_edge(data[i], '19-25', label = "Description")
                exp = domain_knowledge_shap[features[i]]['19-25']
                G.add_node(exp, label = "young")
                G.add_edge(data[i], exp, label = "Value Explanation")
            elif data[i] <= 35:
                G.add_node('26-35', label = '26-35')
                G.add_edge(data[i], '26-35', label = "Description")
                exp = domain_knowledge_shap[features[i]]['26-35']
                G.add_node(exp, label = "young adult")
                G.add_edge(data[i], exp, label = "Value Explanation")
            elif data[i] <= 45:
                G.add_node('36-45', label = '36-45')
                G.add_edge(data[i], '36-45', label = "Description")
                exp = domain_knowledge_shap[features[i]]['36-45']
                G.add_node(exp, label = "middle aged")
                G.add_edge(data[i], exp, label = "Value Explanation")
            elif data[i] <= 55:
                G.add_node('46-55', label = '46-55')
                G.add_edge(data[i], '46-55', label = "Description")
                exp = domain_knowledge_shap[features[i]]['46-55']
                G.add_node(exp, label = "older adult")
                G.add_edge(data[i], exp, label = "Value Explanation")
            elif data[i] <= 65:
                G.add_node('56-65', label = '56-65')
                G.add_edge(data[i], '56-65', label = "Description")
                exp = domain_knowledge_shap[features[i]]['56-65']
                G.add_node(exp, label = "senior")
                G.add_edge(data[i], exp, label = "Value Explanation")
            else:
                G.add_node('65+', label = '65+')
                G.add_edge(data[i], '65+', label = "Description")
                exp = domain_knowledge_shap[features[i]]['65+']
                G.add_node(exp, label = "elderly")
                G.add_edge(data[i], exp, label = "Value Explanation")
        
        if features[i] == "Number_credits":
            if data[i] < 2:
                G.add_node('<2', label = '<2')
                G.add_edge(data[i], '<2', label = "Description")
            else:
                G.add_node('2-4', label = '2-4')
                G.add_edge(data[i], '2-4', label = "Description")
            exp = domain_knowledge_shap[features[i]]["explanation"]
            G.add_node(exp, label = "number of credits")
            G.add_edge(data[i], exp, label = "Value Explanation")
        
        if features[i] == "Number_dependents":
            if data[i] <= 2:
                G.add_node('<=2', label = '<=2')
                G.add_edge(data[i], '<=2', label = "Description")
            exp = domain_knowledge_shap[features[i]]["explanation"]
            G.add_node(exp, label = "number of dependents")
            G.add_edge(data[i], exp, label = "Value Explanation")
        
        if features[i] in protected_attributes:
            protected_attr = protected_attributes[features[i]]["protected_attr"]
            G.add_node(protected_attr, label = protected_attr )
            G.add_edge("Protected attributes", protected_attr, label = "Protected attribute")
    return G


def depth_first_search_shap(graph, node, visited, exp_pred, exp_outcome, exp_protected, idx):
    visited.add(node)
    for neighbor in graph.neighbors(node):
        if graph.has_edge(node, neighbor):
            print(f"Edge from {node} to {neighbor}")
        else: 
            continue
     
        relation = graph.get_edge(node, neighbor).attr['label']
        if relation == "Prediction Meaning":
            exp_pred.append(f"<br> The prediction of '{neighbor}' is based on the following top 5 factors:<ul>")

        if len(graph.out_edges(neighbor)) == 0 and relation != "Protected attribute" and relation != "Protected attributes" and relation != "Prediction Meaning":
            idx[0] += 1
            if idx[0] % 3 == 0:
                exp_outcome.append(f"&nbsp;&nbsp;&#8594;{neighbor} ")
            elif idx[0] % 3 == 1:
                if neighbor == "Personal status and sex" or neighbor == "Age in years" or neighbor == "Foreign worker":
                    exp_outcome.append(f"<li> <strong style='color:red'> {neighbor}: </strong>")
                else:
                    exp_outcome.append(f"<li>{neighbor}: ")
            else:
                exp_outcome.append(f" {neighbor}  </li>")

        if relation == "Protected attributes":
            exp_protected.append(f" Protected attributes: <br><ul>")
        if relation == "Protected attribute":
            exp_protected.append(f"<li> {neighbor} </li>")
       
        if neighbor not in visited:
            depth_first_search_shap(graph, neighbor, visited, exp_pred, exp_outcome, exp_protected, idx)
    return exp_pred, exp_outcome, exp_protected

def shap_nle(graph, root_node):
    explanation = []
    exp_pred, exp_outcome, exp_protected = [], [], []
    idx = [0]
    exp_pred, exp_outcome, exp_protected = depth_first_search_shap(graph, root_node, set(), exp_pred, exp_outcome, exp_protected,idx)
    exp_outcome.append("</ul>")
    
    if len(exp_protected) > 1:
        exp_protected.append("</ul>")
        exp_protected.append("If you believe that protected attributes contributed to a dicriminatory negative prediction, please contact support or a human reviwer.")
        explanation = exp_pred + exp_outcome + exp_protected
    
    else:
        explanation = exp_pred + exp_outcome 
        
    return "".join(explanation)

def neg_shap(loan_data):
    shap_values = generate_feature_importance(loan_data)
    feature_contributions = shap_values.values[0, :]
    feature_names = shap_values.feature_names
    data = shap_values.data[0, :]
    print(data)
    
    
    ### Show top 5 features with negative contributions
    sorted_indices = np.argsort(feature_contributions)
    top_indices = sorted_indices[:5]
    #top_indices = sorted_indices[:1]
    feature_contributions = feature_contributions[top_indices]
    feature_names = [feature_names[i] for i in top_indices]
    data = data[top_indices]
    return feature_names

import gower

def exp_graph_distance(input_data, cases_df, distances):
    G = pgv.AGraph(directed=True)

    G.add_node("Explanation", label ="Explanation")

    G.add_node("Current Case", label = "Current Case")
    G.add_edge("Explanation", "Current Case", label = "Current Case")
    G.add_node("Similar Cases", label = "Similar Cases")
    G.add_edge("Explanation", "Similar Cases", label = "Similar Cases")

    features = input_data.columns


    for i in features:
        G.add_node(f"Crt feature {i}", label = i)
        G.add_edge("Current Case", f"Crt feature {i}", label = "Current Feature")
        feature_descr = domain_knowledge[i]["description"]
        G.add_node(f"crt case {feature_descr}", label = feature_descr)
        G.add_edge(f"Crt feature {i}", f"crt case {feature_descr}", label = "Description")
        
        G.add_node(f"crt case {input_data[i].values[0]}", label = input_data[i].values[0])
        G.add_edge(f"Crt feature {i}", f"crt case {input_data[i].values[0]}", label = "Value of feature")
        if "values" in domain_knowledge[i]:
            data_val = domain_knowledge[i]["values"][input_data[i].values[0]]
            G.add_node(f"crt case {data_val}", label = data_val)
            G.add_edge(f"crt case {input_data[i].values[0]}", f"crt case {data_val}", label = "Description")
            
            

    idx = cases_df.index
    count = 0

    for i in idx:
        G.add_node(i, label = i)
        G.add_edge("Similar Cases", i, label = f"Distance {distances[count]:.2f}")
        count += 1
        for j in features:
            G.add_node(f"{i}: {j}", label = j)
            G.add_edge(i, f"{i}: {j}", label = f"{i} Feature")
            
            feature_descr = domain_knowledge[j]["description"]
            G.add_node(f"{i} {j}: {feature_descr}", label = feature_descr)
            G.add_edge(f"{i}: {j}", f"{i} {j}: {feature_descr}", label = "Description")
            G.add_node(f" {i} {j} : {str(cases_df.loc[i][j])}", label = str(cases_df.loc[i][j]))
            print(cases_df.loc[i][j])
            G.add_edge(f"{i}: {j}", f" {i} {j} : {str(cases_df.loc[i][j])}", label = "Value of feature")
            if "values" in domain_knowledge[j]:
                data_val = domain_knowledge[j]["values"][str(cases_df.loc[i][j])]
                G.add_node(f"{i} {j}: {data_val}", label = data_val)
                G.add_edge(f" {i} {j} : {str(cases_df.loc[i][j])}",f"{i} {j}: {data_val}", label = "Description")
                
                
    return G

def depth_first_search_distance(graph, node, visited, explanation):
    visited.add(node)
    for neighbor in graph.neighbors(node):
        if not graph.has_edge(node, neighbor):
            continue
        relation = graph.get_edge(node, neighbor).attr['label']
        if "Value of feature" in relation:
            feature_name = node.split(":")[-1].strip()
            explanation[feature_name] = {
                "value": neighbor.strip()
            }
        elif "Description" in relation:
            feature_name = node.split(":")[-1].strip()
            if feature_name in explanation:
                explanation[feature_name]["description"] = neighbor.strip()

        
        if neighbor not in visited:
            depth_first_search_distance(graph, neighbor, visited, explanation)
    return explanation


def nle_distance(graph, root_node):
    explanation = []
    explanation = depth_first_search(graph, root_node, set(), explanation)
    print(explanation)
    
    return "".join(explanation)

@login_required
@user_passes_test(is_loan_applicant)
def application_explanation(request):
    if request.method == "POST":
        prediction = request.session.get("prediction")
        loan_data = request.session.get("loan_data")
        
        if not prediction or not loan_data:
            return JsonResponse({'error': 'Please submit a loan application form.'})
        

        user_input = request.POST.get("user_input")


        if user_input == "Remind me my loan status?":
            response = ""
            if prediction[0] == "1":
                response = "Creditworthy"
            else:
                response = "Non-Creditworthy"
            X_PATH = os.path.join(os.path.dirname(__file__), "statlog_german_credit_data")
            feature_names = neg_shap(loan_data)
            german_credit_data = pd.read_csv(X_PATH, index_col=0) 
            german_credit_data.columns = ['Account_status', 'Months', 'Credit_history', 'Purpose', 
                                        'Credit_amount', 'Savings', 'Employment', 'Installment_rate', 
                                        'Personal_status', 'Other_debtors', 'Residence', 'Property', 'Age', 
                                        'Other_installments', 'Housing', 'Number_credits', 'Job', 'Number_dependents', 
                                        'Telephone', 'Foreign_worker', 'target'] 
            X = german_credit_data.loc[:, german_credit_data.columns != "target"]

            y = german_credit_data["target"]
            y = y.replace(2, 0) 
            X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42) # 80% training and 20% test
            
            X_train['target'] = y_train
           
            X_train = X_train[X_train['target'] == int(prediction[0])]
            complete_df = X_train[X_train['target'] == int(prediction[0])]
           
            
            
            X_train = X_train.drop(columns=['target'])

            

            loan_data = pd.DataFrame([loan_data]) 
            loan_data.columns = ['Account_status', 'Months', 'Credit_history', 'Purpose',
                                'Credit_amount', 'Savings', 'Employment', 'Installment_rate', 
                                'Personal_status', 'Other_debtors', 'Residence', 'Property', 'Age', 
                                'Other_installments', 'Housing', 'Number_credits', 'Job', 'Number_dependents', 
                                'Telephone', 'Foreign_worker']
           
            
            data_for_gower = pd.concat([loan_data, X_train], ignore_index=True)
            categorical_cols = ['Account_status', 'Credit_history', 'Purpose', 'Savings', 'Employment',
                                'Personal_status', 'Other_debtors', 'Property', 'Housing', 'Job', 'Telephone', 'Foreign_worker']
            #for col in categorical_cols:
            #        data_for_gower[col] = data_for_gower[col].astype('object')
            #numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

            #scaler = MinMaxScaler()
            #data_for_gower[numeric_cols] = scaler.fit_transform(data_for_gower[numeric_cols])

            data_for_gower = data_for_gower[feature_names]

            # Compute Gower distance matrix (rows x rows)
            gower_dist_matrix = gower.gower_matrix(data_for_gower)

            

            df = loan_data.copy()
            df = df[feature_names]
           
            
            

            print(gower_dist_matrix[0].shape)
            sorted_indices = np.argsort(gower_dist_matrix[0])
            
            top_indices = sorted_indices[:3]

            distances = gower_dist_matrix[0][sorted_indices[1:3]]
           
            for i in top_indices[1:]:
                df = pd.concat([df, X_train.iloc[[i-1]][feature_names]], ignore_index=True)
                
            
            df.index = ["User Input", "Case 1", "Case 2"]

            
            
            for col in df.columns:
                
                for idx in df.index:
                    if "values" in domain_knowledge[col]:
                        df.at[idx, col] = domain_knowledge[col]["values"][df.at[idx, col]]
                df = df.rename(columns={col: domain_knowledge[col]["description"]})
            table_html = df.to_html(index=True, justify='center', escape=False)



            return JsonResponse({'answer': table_html})
        
        if user_input == "How would my loan application status be different?":
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
            #top_indices = sorted_indices[:1]
            feature_contributions = feature_contributions[top_indices]
            feature_names = [feature_names[i] for i in top_indices]
            data = data[top_indices]

            G_shap = shap_graph(feature_names, data, feature_contributions, prediction)
            

            G_shap.node_attr['width'] = '0.3'  # Make nodes smaller
            G_shap.node_attr['height'] = '0.3'  # Make nodes smaller
            G_shap.node_attr['fontsize'] = '12'  # Smaller font for labels
            G_shap.edge_attr['fontsize'] = '12'
            G_shap.node_attr['shape'] = 'box'  # Change node shape to box
            G_shap.layout(prog='dot')
            G_shap.draw('local_shap.png')
            

            response = shap_nle(G_shap, "Explanation")
            


            # response = "The following features contributed to the prediction: "
            return JsonResponse({'answer': response})

        return JsonResponse({'answer': "Sorry, I didn't understand that."})
    return render(request, "application_explanation.html")


domain_knowledge_global_shap = {
    'Account_status': "Reflects the borrower's short-term financial condition and readily available funds. "
                    "A negative account balance indicates a high risk for the borrower to not be able to pay the loan. "
                    "The lack of a checking account adds uncertainty, as the customer's financial situation cannot be verified. "
                    "A strong account status is, by contrast, a positive factor.",

    'Months': "Longer-term loans are riskier for the lender due to uncertainties "
            "about the future financial situation of the borrower. Shorter-term loans are more "
            "favorable, given that the borrower has the capacity to pay the loan.",

    'Credit_history': "Past behavior in repaying debts helps assess future credit risk. "
                    "Previous delays may indicate a pattern that will repeat. Critical account and "
                    "other credits existing suggest limited capabilities of repayment due to multiple existing expenses. "
                    "A history of credits paid in time is a positive factor.",

    'Purpose': "Could indicate the risk of default. Loans for non-essential purposes are "
            "riskier due to uncertainty in future income. Items such as used cars of certain goods depreciate quickly, "
            "which makes it difficult to recover the loan in case of default. Loans for businesses, "
            "retraining and education could also pose risks due to uncertainty in future income, but "
            "may also lead to better financial stability of the borrower.",

    'Credit_amount': "Higher amounts are riskier for lenders, especially if the customer has a bad credit history or low income.",

    'Savings': "Indicates long-term financial stability. Little or no savings suggests higher risk for the lender.",

    'Employment': "Short employment history is a risk factor due to uncertainty in future income. "
                "Long-term employment history is a positive factor.",

    'Installment_rate': "High installment rate means a large proportion of the borrower's income "
                        "is dedicated to loan repayment. This can be risky, due to limited financial stability.",

    'Personal_status': "Reflects family responsibilities and the number of income contributors. "
                    "Family responsibilities may increase financial commitments. Single individuals have fewer obligations, "
                    "but combined with factors like income, debt, and employment history, the lack of financial support may carry risks.",

    'Other_debtors': "Co-applicants and guarantors may reduce risks for lenders due to shared responsibility.", 

    'Residence': "Shorter periods of residence may indicate financial or employment instability. "
                "Longer periods are preferred, depending on the employment opportunities in the local community.",

    'Property': "Property ownership is generally positive, as an indicator of financial stability, "
                "given that the borrower is able to manage property expenses. Similarly, building society savings agreement/ life "
                "insurance are also positive factors. Cars depreciate quickly, so other factors need to be considered when assessing risk. "
                "No asset ownership increases risks for lenders, as they cannot be used as collateral in case of default.",

    'Age': "Younger applicants may have limited financial stability, less credit history, fewer assets, and "
        "limited employment opportunities, which can increase risks. Older applicants are more likely to be financially stable. "
        "However, applicants close to retirement age need to be assessed for their future financial stability.",

    'Other_installments': "Multiple loans are higher risk, due to existing financial commitments. "
                        "Store installment plans may indicate a behavior of over-borrowing. "
                        "No other installment plans is generally a positive factor.",

    'Housing':  "Renting indicates a lack of home ownership and monthly expenses. "
                "Home ownership suggests the presence of an asset that can be used as collateral "
                "in case of default and financial stability. Living for free means fewer financial obligations, "
                "but could also indicate financial instability.",

    'Number_credits': "Multiple existing credits means multiple financial obligations, and may indicate high debt and risk of over-borrowing.",

    'Job': "Unemployed or unskilled workers are higher risk due to lower income levels and limited employment opportunities. "
        "Skilled and highly qualified employees are a positive factor.",

    'Number_dependents': "Higher number of dependents shows increased financial obligations and less flexibility in managing expenses. ",
    'Telephone': "No telephone registration could indicate financial instability and future difficulties in maintaining communication",
    'Foreign_worker': 'Foreign workers may have less credit history, limited job stability and '
                    'employment opportunities, and a lack of local support. ',
}
def shap_graph_global(features, feature_contributions):
    G = pgv.AGraph(directed=True)
    G.add_node("Explanation", label ="Explanation")

    for i in range (len(features)):

        G.add_node(features[i], label = features[i])
        G.add_edge("Explanation", features[i], label = f"Contributes {feature_contributions[i]}")
        feature_descr = domain_knowledge[features[i]]["description"]
        G.add_node(feature_descr, label = feature_descr)
        G.add_edge(features[i], feature_descr, label = "Description")
        # G.add_node(domain_knowledge_global_shap[features[i]], label = domain_knowledge_global_shap[features[i]])
        G.add_node(domain_knowledge_global_shap[features[i]], label = f"{features[i]} explanation")
        G.add_edge(features[i], domain_knowledge_global_shap[features[i]], label = "Global explanation")
    
    return G

def depth_first_search_shap_global(graph, node, visited, explanation, idx = 0):
    visited.add(node)
    for neighbor in graph.neighbors(node):
        if graph.has_edge(node, neighbor):
            print(f"Edge from {node} to {neighbor}")
        else: 
            continue
        
        if len(graph.out_edges(neighbor)) == 0:
            idx += 1
            if idx % 2 == 1:
                explanation.append(f"<li>{neighbor}: ")
            if idx % 2 == 0:
                explanation.append(f"&nbsp;&nbsp;&#8594;{neighbor} </li>\n")
            
        if neighbor not in visited:
            depth_first_search_shap_global(graph, neighbor, visited, explanation, idx)
    return explanation

def shap_nle_global(graph, root_node):
    explanation = []
    explanation = depth_first_search_shap_global(graph, root_node, set(), explanation)
    explanation.insert(0, "\n <strong>Top 10 most important features:\n </strong><ol>")
    explanation.append("</ol>")
    print(explanation)
    
    return "".join(explanation)

fairness_graph_domain = {
    ## Features
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

    "Age_group":{
        "description": "Age in years",
        "values": {
            "19-25": "young (under 25)",
            "26-35": "26 - 35",
            "36-45": "36 - 45",
            "46-55": "46 - 55",
            "56-65": "56 - 65",
            "66-75": "65+",
        }
    },

    "Foreign_worker":{
        "description": "Foreign worker",
        "values": {
            "A201": "yes",
            "A202": "no",
        }
    },

    "Male_female":{
        "description": "Sex (M/F)",
        "values": {
            "Male": "male",
            "Female": "female",
        }
    },

    ## Fairness metrics
    "Demographic Parity Ratio":{
        "description": "A ratio of 1 means equal chances of selection across groups",
        
    },
    "Equalized Odds Ratio": {
        "description": "A ratio of 1 means equal performance across groups",
    },
    "Equal Opportunity Ratio":{
        "description": "A ratio of 1 means equal performance across groups for positive predictions",
    }

}

common_fairness_graph_domain = {
    "Demographic Parity":{
        "description": "Checks if positive predictions rates are equal across groups.",
        "Demographic Parity Ratio": "A ratio of 1 means equal chances of selection across groups",
    },
    "Equalized Odds":{
        "description": "Checks if prediction errors are equal across groups.",
        "Equalized Odds Ratio": "A ratio of 1 means equal true positive, true negative, false positive, and false negative rates across groups",
    },
    "Equal Opportunity":{
        "description": "Measures model performance for positive predictions across groups.",
        "Equal Opportunity Ratio": "A ratio of 1 means equal true positive rates across groups",
    },
}


def common_fairness_graph(common_fairness_dict):
    G = pgv.AGraph(directed=True)

    G.add_node("Explanation", label ="Explanation")
    G.add_node("Demographic Parity", label = "Demographic Parity")
    G.add_edge("Explanation", "Demographic Parity", label = "Common Metrics")
    G.add_node("Equalized Odds", label = "Equalized Odds")
    G.add_edge("Explanation", "Equalized Odds", label = "Common Metrics")
    G.add_node("Equal Opportunity", label = "Equal Opportunity")
    G.add_edge("Explanation", "Equal Opportunity", label = "Common Metrics")

    G.add_node(common_fairness_graph_domain["Demographic Parity"]["description"], label = common_fairness_graph_domain["Demographic Parity"]["description"])
    G.add_edge("Demographic Parity", common_fairness_graph_domain["Demographic Parity"]["description"], label = "Description")
    G.add_node(common_fairness_graph_domain["Equalized Odds"]["description"], label = common_fairness_graph_domain["Equalized Odds"]["description"])
    G.add_edge("Equalized Odds", common_fairness_graph_domain["Equalized Odds"]["description"], label = "Description")
    G.add_node(common_fairness_graph_domain["Equal Opportunity"]["description"], label = common_fairness_graph_domain["Equal Opportunity"]["description"])
    G.add_edge("Equal Opportunity", common_fairness_graph_domain["Equal Opportunity"]["description"], label = "Description")

    G.add_node("Demographic Parity Ratio", label = "Demographic Parity Ratio")
    G.add_edge("Demographic Parity", "Demographic Parity Ratio", label = "Ratio")
    G.add_node("Equalized Odds Ratio", label = "Equalized Odds Ratio")
    G.add_edge("Equalized Odds", "Equalized Odds Ratio", label = "Ratio")
    G.add_node("Equal Opportunity Ratio", label = "Equal Opportunity Ratio")
    G.add_edge("Equal Opportunity", "Equal Opportunity Ratio", label = "Ratio")
    G.add_node(common_fairness_graph_domain["Demographic Parity"]["Demographic Parity Ratio"], label = common_fairness_graph_domain["Demographic Parity"]["Demographic Parity Ratio"])
    G.add_edge("Demographic Parity Ratio", common_fairness_graph_domain["Demographic Parity"]["Demographic Parity Ratio"], label = "Description")
    G.add_node(common_fairness_graph_domain["Equalized Odds"]["Equalized Odds Ratio"], label = common_fairness_graph_domain["Equalized Odds"]["Equalized Odds Ratio"])
    G.add_edge("Equalized Odds Ratio", common_fairness_graph_domain["Equalized Odds"]["Equalized Odds Ratio"], label = "Description")
    G.add_node(common_fairness_graph_domain["Equal Opportunity"]["Equal Opportunity Ratio"], label = common_fairness_graph_domain["Equal Opportunity"]["Equal Opportunity Ratio"])
    G.add_edge("Equal Opportunity Ratio", common_fairness_graph_domain["Equal Opportunity"]["Equal Opportunity Ratio"], label = "Description")

    for feature, (dp_ratio, eo_ratio, eopp_ratio) in common_fairness_dict.items():
        G.add_node(f"Demographic parity ratio {feature}", label = feature)
        G.add_edge("Demographic Parity Ratio", f"Demographic parity ratio {feature}", label = "Feature")
        feature_descr = fairness_graph_domain[feature]["description"]
        G.add_node(f"Demographic parity ratio {feature_descr}", label = feature_descr)
        G.add_edge(f"Demographic parity ratio {feature}", f"Demographic parity ratio {feature_descr}", label = "Feature Description")



        G.add_node(f"Demographic Parity: {dp_ratio}", label = f"{round(dp_ratio,2)}")
        G.add_edge(f"Demographic parity ratio {feature}", f"Demographic Parity: {dp_ratio}", label = "Value of Metric")
        G.add_node(f"Equalized odds ratio {feature}", label = feature)
        G.add_edge("Equalized Odds Ratio", f"Equalized odds ratio {feature}", label = "Feature")
        feature_descr = fairness_graph_domain[feature]["description"]
        G.add_node(f"Equalized odds ratio {feature_descr}", label = feature_descr)
        G.add_edge(f"Equalized odds ratio {feature}", f"Equalized odds ratio {feature_descr}", label = "Feature Description")

        G.add_node(f"Equalized Odds: {eo_ratio}", label = f"{round(eo_ratio,2)}")
        G.add_edge(f"Equalized odds ratio {feature}", f"Equalized Odds: {eo_ratio}", label = "Value of Metric")
        G.add_node(f"Equal opportunity ratio {feature}", label = feature)
        G.add_edge("Equal Opportunity Ratio", f"Equal opportunity ratio {feature}", label = "Feature")
        feature_descr = fairness_graph_domain[feature]["description"]
        G.add_node(f"Equal opportunity ratio {feature_descr}", label = feature_descr)
        G.add_edge(f"Equal opportunity ratio {feature}", f"Equal opportunity ratio {feature_descr}", label = "Feature Description")
        G.add_node(f"Equal Opportunity: {eopp_ratio}", label = f"{round(eopp_ratio,2)}")
        G.add_edge(f"Equal opportunity ratio {feature}", f"Equal Opportunity: {eopp_ratio}", label = "Value of Metric")

    return G

def depth_first_search_common_fairness(graph, node, visited, explanation):
    visited.add(node)
    for neighbor in graph.neighbors(node):
        if graph.has_edge(node, neighbor):
            print(f"Edge from {node} to {neighbor}")
        else: 
            continue
        relation = graph.get_edge(node, neighbor).attr['label']
        if relation == "Common Metrics":
            explanation.append( f"<strong>\n\n {neighbor}: </strong>")
        if relation == "Description":
            explanation.append(f"&nbsp;{neighbor}\n")
        if relation == "Ratio":
            explanation.append(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {neighbor}: ")
        if relation == "Feature Description":
            label = graph.get_node(neighbor).attr['label']
            explanation.append(f"&nbsp;&nbsp;&nbsp;&nbsp;&#8594; {label}: ")

        
    
        
        
        
            
        if relation == "Value of Metric":
            label = graph.get_node(neighbor).attr['label']
            
            if float(label) < 0.5:
                explanation.append(f"<strong style='color:red'>&nbsp;&nbsp;{label} </strong>\n")
            else:
                explanation.append(f"&nbsp;&nbsp;{label}\n")

        
            
        
     
        if neighbor not in visited:
            depth_first_search_common_fairness(graph, neighbor, visited, explanation)
        
    return explanation

def common_fairness_nle(graph, root_node):
    explanation = []
    explanation = depth_first_search_common_fairness(graph, root_node, set(), explanation)
   
    
    return "".join(explanation)

def fairness_graph(overall, by_group, group_min, group_max, diff, ratio, overall_diff, overall_ratio, fairness_metrics, feature):

    
    G = pgv.AGraph(directed=True)

    G.add_node("Explanation", label ="Explanation")
    G.add_node(feature, label = feature)
    G.add_edge("Explanation", feature, label = feature)
    feature_descr = fairness_graph_domain[feature]["description"]
    G.add_node(feature_descr, label = feature_descr)
    G.add_edge(feature, feature_descr, label = "Feature Description")

    

    for col, values in by_group.items():
        if col == "recall/ true_positive_rate":
            G.add_node(col, label = "recall/ true positive rate")
        elif col == "selection_rate":
            G.add_node(col, label = "selection rate")
        elif col == "false_positive_rate":
            G.add_node(col, label = "false positive rate")
        elif col == "false_negative_rate":
            G.add_node(col, label = "false negative rate")
        else:
            G.add_node(col, label = col)
        G.add_edge(feature, col, label = "Metric")

        overall_metric = overall.loc[col]
        G.add_node(f"Overall {col}: {overall_metric}", label = f"Overall: {round(overall_metric,2)}")
        G.add_edge(col, f"Overall {col}: {overall_metric}", label = "Overall Metric")
       

        for index, value in values.items():
            if "values" in fairness_graph_domain[feature]:
                index_descr = fairness_graph_domain[feature]["values"][index]
                G.add_node(f"{col} {index_descr}", label = index_descr)
                G.add_edge(f"{col} {index}", f"{col} {index_descr}", label = "Description")
            G.add_edge(f"{col} {index}", f"{col} {index} {value}", label = "Value of Metric")
            G.add_node(f"{col} {index}", label = index)
            G.add_edge(col, f"{col} {index}", label = "Group")
            G.add_node(f"{col} {index} {value}", label = round(value,2))
            

            if value == group_max[col]:
                G.add_node(f"{col} {index} Group Max", label = "Group Max")
                G.add_edge(f"{col} {index}", f"{col} {index} Group Max", label = "Group Max")
            if value == group_min[col]:
                G.add_node(f"{col} {index} Group Min", label = "Group Min")
                G.add_edge(f"{col} {index}", f"{col} {index} Group Min", label = "Group Min")
            
   

    return G

def depth_first_search_fairness(graph, node, visited, explanation):
    visited.add(node)
    for neighbor in graph.neighbors(node):
        if graph.has_edge(node, neighbor):
            print(f"Edge from {node} to {neighbor}")
        else: 
            continue
        relation = graph.get_edge(node, neighbor).attr['label']
        if relation == "Feature Description":
            explanation.insert(0, f"<strong>\n\n\n\n Feature {neighbor}: </strong>")
        if relation == "Metric":
            label = graph.get_node(neighbor).attr['label']
            explanation.append(f"<i>\n\n&nbsp;Metrics for {label}:\n</i>")
            
        if relation == "Overall Metric":
            label = graph.get_node(neighbor).attr['label']
            explanation.append(f"&nbsp;&nbsp;{label}\n")

        if relation == "Overall Metric":
            
            explanation.append("&nbsp;&nbsp;&nbsp;Group metrics for each value of feature")
       
            
        if relation == "Description":
            label = graph.get_node(neighbor).attr['label']
            explanation.append(f"\n &nbsp;&nbsp;&nbsp;&nbsp;&#8594;Group {label}: ")
        if relation == "Value of Metric":
            label = graph.get_node(neighbor).attr['label']
            explanation.append(f"{label}")
        if relation == "Group Max":
            explanation.append(f" <strong style='color:red'> This group has the highest value! </strong>")
        if relation == "Group Min":
            explanation.append(f" <strong style='color:red'> This group has the lowest value!</strong>")
        
        if relation == "Fairness metrics":
            explanation.append(f"<i>\n\n{neighbor}: \n</i>")
            
        
            
        if neighbor not in visited:
            depth_first_search_fairness(graph, neighbor, visited, explanation)
    return explanation

def fairness_nle(graph, root_node):
    explanation = []
    explanation = depth_first_search_fairness(graph, root_node, set(), explanation)
    
    return "".join(explanation)


shap_fairness_domain = {
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

    "Age_group":{
        "description": "Age in years",
        "values": {
            "19-25": "young (under 25)",
            "26-35": "26-35",
            "36-45": "36-45",
            "46-55": "46-55",
            "56-65": "56-65",
            "66-75": "65+",
        }
    },

    "Foreign_worker":{
        "description": "Foreign worker",
        "values": {
            "A201": "yes",
            "A202": "no",
        }
    },

    "Male_female":{
        "description": "Sex (M/F)",
        "values": {
            "Male": "male",
            "Female": "female",
        }
    },

}
def shap_graph_fairness(shap_data, feature):
    G = pgv.AGraph(directed=True)
    G.add_node("Explanation", label ="Explanation")
    G.add_node(feature, label = feature)
    G.add_edge("Explanation", feature, label = "Feature")
    feature_descr = shap_fairness_domain[feature]["description"]
    G.add_node(feature_descr, label = feature_descr)
    G.add_edge(feature, feature_descr, label = "Feature Description")

    for group, (feature_names, feature_contributions) in shap_data.items():
        G.add_node(group, label = group)
        G.add_edge(feature, group, label = "Group")

        group_descr = shap_fairness_domain[feature]["values"][group]
        G.add_node(group_descr, label = group_descr)
        G.add_edge(group, group_descr, label = "Group Description")

        for i in range (len(feature_names)):
            G.add_node(f"{group} {feature_names[i]}", label = feature_names[i])
            G.add_edge(group, f"{group} {feature_names[i]}", label = f"Contributes {feature_contributions[i]:.2f}")
            
            feature_descr = domain_knowledge[feature_names[i]]["description"]
            
            G.add_node(f"{group} {feature_descr}", label = feature_descr)
            G.add_edge(f"{group} {feature_names[i]}", f"{group} {feature_descr}", label = "Description")

           
    
    return G
protected_attributes_shap_fairness = ["Personal status and sex", "Age in years", "Foreign worker"]

def depth_first_search_shap_fairness(graph, node, visited, explanation, idx):
    visited.add(node)
    for neighbor in graph.neighbors(node):
        if graph.has_edge(node, neighbor):
            print(f"Edge from {node} to {neighbor}")
        else: 
            continue
        relation = graph.get_edge(node, neighbor).attr['label']
        if relation == "Feature Description":
            explanation.insert(0, f"\n<strong>Top 10 most important features for {neighbor}:</strong> \n")
        if relation == "Group Description":
            explanation.append(f"\n<i>&nbsp;Group {neighbor}:</i> \n")
            idx[0] = 0
        if len(graph.out_edges(neighbor)) == 0 and relation != "Feature Description" and relation != "Group Description":
            idx[0] += 1
            #print(idx)
            label = graph.get_node(neighbor).attr['label']
            if label in protected_attributes_shap_fairness:
                explanation.append(f"{idx}. <strong style='color:red'>&nbsp;&nbsp;&#8594{label} </strong>\n")
            elif relation == "Negative contribution":
                explanation.append(f"{idx}. <strong style='color:red'>&nbsp;&nbsp;&#8594{label} </strong>\n")
            elif relation == "Positive contribution":
                explanation.append(f"{idx}. <strong style='color:green'>&nbsp;&nbsp;&#8594{label} </strong>\n")
            else:
                explanation.append(f"{idx}. &nbsp;&nbsp;&#8594{label} \n")
            
        
        if neighbor not in visited:
            depth_first_search_shap_fairness(graph, neighbor, visited, explanation, idx)
    return explanation


def fairness_shap_nle(graph, root_node):
    explanation = []
    idx = [0]
    explanation = depth_first_search_shap_fairness(graph, root_node, set(), explanation, idx)
    #print(explanation)
    
    
    return "".join(explanation)

def is_employee(user):
    return user.groups.filter(name='employee').exists()


@login_required
@user_passes_test(is_employee)
def model_explanation(request):
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
            y = german_credit_data["target"]
            y = y.replace(2, 0) 
            X_train, _, _, _  = train_test_split(X, y, test_size=0.2, random_state=42) 
            explainer = shap.Explainer(model1)
            
            shap_values = explainer(X)
            
            mean_abs_values = np.abs(shap_values.values).mean(axis=0) 
            #print(mean_abs_values)
            top_10_indices = np.argsort(-mean_abs_values)[:10]
            #print(top_10_indices)
            #top_10_indices = np.argsort(-mean_abs_values)[:2]
            feature_names = np.array(shap_values.feature_names)
            feature_names = feature_names[top_10_indices]
            

            
            #print(feature_names)
            # shap.plots.bar(shap_values)
            
            

            G = shap_graph_global(feature_names, mean_abs_values[top_10_indices])
            
            
            
            response = shap_nle_global(G, "Explanation")

            # shap.summary_plot(shap_values, X)
            return JsonResponse({'answer': response})
        
        
        if user_input == "I want to see common fairness metrics" or user_input == "I want to see performance-based fairness":
            X_PATH = os.path.join(os.path.dirname(__file__), "statlog_german_credit_data")

            german_credit_data = pd.read_csv(X_PATH, index_col=0) 
            german_credit_data.columns = ['Account_status', 'Months', 'Credit_history', 'Purpose', 
                                        'Credit_amount', 'Savings', 'Employment', 'Installment_rate', 
                                        'Personal_status', 'Other_debtors', 'Residence', 'Property', 'Age', 
                                        'Other_installments', 'Housing', 'Number_credits', 'Job', 'Number_dependents', 
                                        'Telephone', 'Foreign_worker', 'target'] 
            X = german_credit_data.loc[:, german_credit_data.columns != "target"]

            y = german_credit_data["target"]
            y = y.replace(2, 0) 
            X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42) # 80% training and 20% test

            y_pred = model1.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            
            
            response = ""


            personal_status = X_test["Personal_status"]
            
            
            
            metric_frame_personal_status = MetricFrame(
                metrics={"accuracy": accuracy_score, 
                        
                         "recall/ true_positive_rate": true_positive_rate,
                        
                         "selection_rate": selection_rate, 
                         "false_positive_rate": false_positive_rate, 
                         "false_negative_rate": false_negative_rate,
                         
                         "count": count, 
                         },
                y_true=y_test,
                y_pred=y_pred,
                sensitive_features=personal_status,
            )
            
            
            overall_personal_status = metric_frame_personal_status.overall
            #print(f"Overall:\n{overall_personal_status}")

            by_group_personal_status = metric_frame_personal_status.by_group
            #print(f"By group:\n{by_group_personal_status}")

            group_min_personal_status = metric_frame_personal_status.group_min()
            print(f"Group min:\n{group_min_personal_status}")

            group_max_personal_status = metric_frame_personal_status.group_max()
            #print(f"Group max:\n{group_max_personal_status}")

            diff_personal_status = metric_frame_personal_status.difference()

            ratio_personal_status = metric_frame_personal_status.ratio()

            overall_diff_personal_status = metric_frame_personal_status.difference(method = "to_overall")

            overall_ratio_personal_status = metric_frame_personal_status.ratio(method = "to_overall")

            

            
            demographic_parity_ratio_personal_status  = demographic_parity_ratio(y_true = y_test,
                                                               y_pred = y_pred,
                                                               sensitive_features=personal_status,)
          
            equalized_odds_ratio_personal_status  = equalized_odds_ratio(y_true = y_test,
                                                         y_pred = y_pred,
                                                         sensitive_features=personal_status,)
       
            equal_opportunity_ratio_personal_status  = equal_opportunity_ratio(y_true = y_test,
                                                                y_pred = y_pred,
                                                                sensitive_features=personal_status,)
            
           

            fairness_metrics_personal_status = {
                #"Demographic Parity Difference": demographic_parity_diff_personal_status,
                "Demographic Parity Ratio": demographic_parity_ratio_personal_status,
                #"Equalized Odds Difference": equalized_odds_diff_personal_status,
                "Equalized Odds Ratio": equalized_odds_ratio_personal_status,
                #"Equal Opportunity Difference": equal_opportunity_diff_personal_status,
                "Equal Opportunity Ratio": equal_opportunity_ratio_personal_status,
            }
            

            G_personal_status = fairness_graph(overall_personal_status, by_group_personal_status, group_min_personal_status, 
                                               group_max_personal_status, diff_personal_status, ratio_personal_status, overall_diff_personal_status, 
                                               overall_ratio_personal_status, fairness_metrics_personal_status, "Personal_status")
            
            
            
            
            
            

            #response_personal_status = fairness_nle(G_personal_status, "Explanation")


            
            
            
            X_bin = X_test.copy()
            
            X_bin["Age_group"] = pd.cut(X_bin["Age"], bins=[19, 25, 35, 45, 55, 65, 75], labels=["19-25", "26-35", "36-45", "46-55", "56-65", "66-75"])
            print(X_bin[["Age", "Age_group"]].head(10))
            metric_frame_age_bin = MetricFrame(
                metrics={"accuracy": accuracy_score, 
                         #"precision": precision_score, 
                         "recall/ true_positive_rate": true_positive_rate,
                         #"f1": f1_score,
                         "selection_rate": selection_rate, 
                          
                         "false_positive_rate": false_positive_rate, 
                         "false_negative_rate": false_negative_rate,
                         #"true_positive_rate": true_positive_rate,
                         #"true_negative_rate": true_negative_rate,  
                         "count": count,
                         },
                y_true=y_test,
                y_pred=y_pred,
                sensitive_features=X_bin["Age_group"],
            )
            
            overall_age = metric_frame_age_bin.overall
            by_group_age = metric_frame_age_bin.by_group
            group_min_age = metric_frame_age_bin.group_min()
            group_max_age = metric_frame_age_bin.group_max()
            diff_age = metric_frame_age_bin.difference()
            ratio_age = metric_frame_age_bin.ratio()
            overall_diff_age = metric_frame_age_bin.difference(method = "to_overall")
            overall_ratio_age = metric_frame_age_bin.ratio(method = "to_overall")
            
            demographic_parity_ratio_age  = demographic_parity_ratio(y_true = y_test,
                                                               y_pred = y_pred,
                                                               sensitive_features=X_bin["Age_group"],)
            
            equalized_odds_ratio_age  = equalized_odds_ratio(y_true = y_test,
                                                         y_pred = y_pred,
                                                         sensitive_features=X_bin["Age_group"],)
           
            equal_opportunity_ratio_age  = equal_opportunity_ratio(y_true = y_test, 
                                                                y_pred = y_pred,
                                                                sensitive_features=X_bin["Age_group"],)
            fairness_metrics_age = {
                #"Demographic Parity Difference": demographic_parity_diff_age,
                "Demographic Parity Ratio": demographic_parity_ratio_age,
                #"Equalized Odds Difference": equalized_odds_diff_age,
                "Equalized Odds Ratio": equalized_odds_ratio_age,
                #"Equal Opportunity Difference": equal_opportunity_diff_age,
                "Equal Opportunity Ratio": equal_opportunity_ratio_age,
            }

            #print(metric_frame_age_bin.by_group)
            G_age_bin = fairness_graph(overall_age, by_group_age, group_min_age, group_max_age, diff_age, 
                                       ratio_age, overall_age, overall_ratio_age, fairness_metrics_age, "Age_group")
            G_age_bin.draw("age_bin.png", format="png", prog="dot")
           
            
            #response_age = fairness_nle(G_age_bin, "Explanation")
            response = ""
            foreign_worker = X_test["Foreign_worker"]
            metric_frame_foreign_worker = MetricFrame(
                metrics={"accuracy": accuracy_score, 
                         #"precision": precision_score, 
                         "recall/ true_positive_rate": true_positive_rate,
                         #"f1": f1_score,
                         "selection_rate": selection_rate, 
                         
                         "false_positive_rate": false_positive_rate, 
                         "false_negative_rate": false_negative_rate,
                         #"true_positive_rate": true_positive_rate,
                         #"true_negative_rate": true_negative_rate,  
                         "count": count, 
                         },
                    
                y_true=y_test,
                y_pred=y_pred,
                sensitive_features=foreign_worker,
            )
            
            overall_foreign_worker = metric_frame_foreign_worker.overall
            by_group_foreign_worker = metric_frame_foreign_worker.by_group
            print(f"By group:\n{by_group_foreign_worker}")
            group_min_foreign_worker = metric_frame_foreign_worker.group_min()
            group_max_foreign_worker = metric_frame_foreign_worker.group_max()  
            diff_foreign_worker = metric_frame_foreign_worker.difference()
            ratio_foreign_worker = metric_frame_foreign_worker.ratio()
            overall_diff_foreign_worker = metric_frame_foreign_worker.difference(method = "to_overall")
            overall_ratio_foreign_worker = metric_frame_foreign_worker.ratio(method = "to_overall")
           
            demographic_parity_ratio_foreign_worker  = demographic_parity_ratio(y_true = y_test,
                                                               y_pred = y_pred,
                                                               sensitive_features=foreign_worker,)
            
            equalized_odds_ratio_foreign_worker  = equalized_odds_ratio(y_true = y_test,
                                                         y_pred = y_pred,
                                                         sensitive_features=foreign_worker,)
          
            equal_opportunity_ratio_foreign_worker  = equal_opportunity_ratio(y_true = y_test,
                                                                y_pred = y_pred,
                                                                sensitive_features=foreign_worker,)
            fairness_metrics_foreign_worker = { 
                #"Demographic Parity Difference": demographic_parity_diff_foreign_worker,
                "Demographic Parity Ratio": demographic_parity_ratio_foreign_worker,
                #"Equalized Odds Difference": equalized_odds_diff_foreign_worker,
                "Equalized Odds Ratio": equalized_odds_ratio_foreign_worker,
                #"Equal Opportunity Difference": equal_opportunity_diff_foreign_worker,
                "Equal Opportunity Ratio": equal_opportunity_ratio_foreign_worker,
            }

            #print(metric_frame_foreign_worker.by_group)
            #print(metric_frame_foreign_worker.overall)
            G_foreign_worker = fairness_graph(overall_foreign_worker, by_group_foreign_worker, group_min_foreign_worker, group_max_foreign_worker, diff_foreign_worker,
                                       ratio_foreign_worker, overall_diff_foreign_worker, overall_ratio_foreign_worker, fairness_metrics_foreign_worker, "Foreign_worker")

            #response_foreign_worker = fairness_nle(G_foreign_worker, "Explanation")
            

            personal_status_mapping = {
                "A91": "Male",
                "A92": "Female",
                "A93": "Male",
                "A94": "Male",
                "A95": "Female", 
            }
            X_map = X_test.copy()
            X_map["male_female"] = X_map['Personal_status'].map(personal_status_mapping)
            metric_frame_male_female = MetricFrame(
                metrics={#"accuracy": accuracy_score, 
                         #"precision": precision_score, 
                         #"recall/ true_positive_rate": true_positive_rate,
                         #"f1": f1_score,
                         #"selection_rate": selection_rate, 
                         
                         #"false_positive_rate": false_positive_rate, 
                         #"false_negative_rate": false_negative_rate,
                         #"true_positive_rate": true_positive_rate,
                         #"true_negative_rate": true_negative_rate,  
                         "count": count, 
                         },
                    
                y_true=y_test,
                y_pred=y_pred,
                sensitive_features=X_map["male_female"],
            )
            
            print(metric_frame_male_female.by_group)

            overall_male_female = metric_frame_male_female.overall
            by_group_male_female = metric_frame_male_female.by_group
            group_min_male_female = metric_frame_male_female.group_min()
            group_max_male_female= metric_frame_male_female.group_max()  
            diff_male_female = metric_frame_male_female.difference()
            ratio_male_female = metric_frame_male_female.ratio()
            overall_diff_male_female= metric_frame_male_female.difference(method = "to_overall")
            overall_ratio_male_female= metric_frame_male_female.ratio(method = "to_overall")

            

            
            demographic_parity_ratio_male_female = demographic_parity_ratio(y_true = y_test,
                                                                            y_pred = y_pred,
                                                                            sensitive_features = X_map["male_female"],)
            
            
            equalized_odds_ratio_male_female  = equalized_odds_ratio(y_true = y_test,
                                                         y_pred = y_pred,
                                                         sensitive_features=X_map["male_female"])
           
            equal_opportunity_ratio_male_female = equal_opportunity_ratio(y_true = y_test,
                                                                y_pred = y_pred,
                                                                sensitive_features=X_map["male_female"],)
            
            fairness_metrics_male_female = { 
                #"Demographic Parity Difference": demographic_parity_diff_foreign_worker,
                "Demographic Parity Ratio": demographic_parity_ratio_male_female ,
                #"Equalized Odds Difference": equalized_odds_diff_foreign_worker,
                "Equalized Odds Ratio": equalized_odds_ratio_male_female ,
                #"Equal Opportunity Difference": equal_opportunity_diff_foreign_worker,
                "Equal Opportunity Ratio": equal_opportunity_ratio_male_female ,
            }

            G_male_female = fairness_graph(overall_male_female, by_group_male_female, group_min_male_female, group_max_male_female,
                                           diff_male_female, ratio_male_female, overall_diff_male_female, overall_ratio_male_female,
                                           fairness_metrics_male_female, "Male_female") 
            G_male_female.node_attr['width'] = '0.3'  # Make nodes smaller
            G_male_female.node_attr['height'] = '0.3'  # Make nodes smaller
            G_male_female.node_attr['fontsize'] = '12'  # Smaller font for labels
            G_male_female.edge_attr['fontsize'] = '12'
            G_male_female.node_attr['shape'] = 'box'  # Change node shape to box
            G_male_female.layout(prog='dot')

            G_male_female.draw("male_female.png", format="png", prog="dot") 
           
            

            #response_male_female = fairness_nle(G_male_female, "Explanation")                                            
            common_fairness_dict = {
                "Personal_status": (demographic_parity_ratio_personal_status, equalized_odds_ratio_personal_status, equal_opportunity_ratio_personal_status),
                "Age_group": (demographic_parity_ratio_age, equalized_odds_ratio_age, equal_opportunity_ratio_age),
                "Foreign_worker": (demographic_parity_ratio_foreign_worker, equalized_odds_ratio_foreign_worker, equal_opportunity_ratio_foreign_worker),
                "Male_female": (demographic_parity_ratio_male_female, equalized_odds_ratio_male_female, equal_opportunity_ratio_male_female),
            }

            #response = response_personal_status + response_male_female + response_age + response_foreign_worker
            if user_input == "I want to see common fairness metrics":
                G = common_fairness_graph(common_fairness_dict)
                G.draw("common_fairness.png", format="png", prog="dot")
                response = common_fairness_nle(G, "Explanation")

            if user_input == "I want to see performance-based fairness":
                response_personal_status = fairness_nle(G_personal_status, "Explanation")
                response_age = fairness_nle(G_age_bin, "Explanation")
                response_foreign_worker = fairness_nle(G_foreign_worker, "Explanation")
                response_male_female = fairness_nle(G_male_female, "Explanation")
                response = response_personal_status + response_age + response_foreign_worker + response_male_female
               
            
            return JsonResponse({'answer': response})
           
            #return JsonResponse({'answer': response})
        if user_input == "I want to see feature importance for sensitive features":
            X_PATH = os.path.join(os.path.dirname(__file__), "statlog_german_credit_data")

            german_credit_data = pd.read_csv(X_PATH, index_col=0) 
            german_credit_data.columns = ['Account_status', 'Months', 'Credit_history', 'Purpose', 
                                        'Credit_amount', 'Savings', 'Employment', 'Installment_rate', 
                                        'Personal_status', 'Other_debtors', 'Residence', 'Property', 'Age', 
                                        'Other_installments', 'Housing', 'Number_credits', 'Job', 'Number_dependents', 
                                        'Telephone', 'Foreign_worker', 'target'] 
            X = german_credit_data.loc[:, german_credit_data.columns != "target"]

            y = german_credit_data["target"]
            y = y.replace(2, 0) 
            X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42) 

            X_filtered = X_train.copy()
            explainer = shap.Explainer(model1)


            # Personal status and sex
            X_filtered_male_91 = X_filtered[X_filtered["Personal_status"] == "A91"]
            shap_values_male_91 = explainer(X_filtered_male_91)
            mean_abs_values_male_91 = np.abs(shap_values_male_91.values).mean(axis=0)
            top_10_indices_male_91 = np.argsort(-mean_abs_values_male_91)[:10]
            feature_names_male_91 = np.array(shap_values_male_91.feature_names)
            feature_names_male_91 = feature_names_male_91[top_10_indices_male_91]

            X_filtered_female_92 = X_filtered[X_filtered["Personal_status"] == "A92"]
            shap_values_female_92 = explainer(X_filtered_female_92)
            # shap.plots.bar(shap_values_female_92)
            mean_abs_values_female_92 = np.abs(shap_values_female_92.values).mean(axis=0)
            top_10_indices_female_92 = np.argsort(-mean_abs_values_female_92)[:10]
            feature_names_female_92 = np.array(shap_values_female_92.feature_names)
            feature_names_female_92 = feature_names_female_92[top_10_indices_female_92]

            X_filtered_male_93 = X_filtered[X_filtered["Personal_status"] == "A93"]
            shap_values_male_93 = explainer(X_filtered_male_93)
            #shap.plots.bar(shap_values_male_93)
            mean_abs_values_male_93 = np.abs(shap_values_male_93.values).mean(axis=0)
            top_10_indices_male_93 = np.argsort(-mean_abs_values_male_93)[:10]
            feature_names_male_93 = np.array(shap_values_male_93.feature_names)
            feature_names_male_93 = feature_names_male_93[top_10_indices_male_93]

            X_filtered_male_94 = X_filtered[X_filtered["Personal_status"] == "A94"]
            shap_values_male_94 = explainer(X_filtered_male_94)
            #shap.plots.bar(shap_values_male_94)
            mean_abs_values_male_94 = np.abs(shap_values_male_94.values).mean(axis=0)
            top_10_indices_male_94 = np.argsort(-mean_abs_values_male_94)[:10]
            feature_names_male_94 = np.array(shap_values_male_94.feature_names)
            feature_names_male_94 = feature_names_male_94[top_10_indices_male_94]

            shap_data_personal_status = {
                "A91": (feature_names_male_91, mean_abs_values_male_91[top_10_indices_male_91]),
                "A92": (feature_names_female_92, mean_abs_values_female_92[top_10_indices_female_92]),
                "A93": (feature_names_male_93, mean_abs_values_male_93[top_10_indices_male_93]),
                "A94": (feature_names_male_94, mean_abs_values_male_94[top_10_indices_male_94]),
            }
            '''
            shap_values = explainer.shap_values(X_filtered)
            mask_91 = X_filtered["Personal_status"] == "A91"
            mask_92 = X_filtered["Personal_status"] == "A92"
            mask_93 = X_filtered["Personal_status"] == "A93"
            mask_94 = X_filtered["Personal_status"] == "A94"
            shap_sum = shap_values.sum(1)
            mean_91 = shap_sum[mask_91].mean()
            mean_92 = shap_sum[mask_92].mean()
            mean_93 = shap_sum[mask_93].mean()
            mean_94 = shap_sum[mask_94].mean()
            min_val = min(mean_91, mean_92, mean_93, mean_94)
            max_val = max(mean_91, mean_92, mean_93, mean_94)
            ratio = min_val/max_val 

            print("Ratio")
            print(ratio)
            print(min_val)
            print(mean_91)
            '''



            # FOREIGN WORKER

            X_filtered_fk = X_filtered[X_filtered["Foreign_worker"] == "A201"]
            shap_values_fk = explainer(X_filtered_fk)
            mean_abs_values_fk = np.abs(shap_values_fk.values).mean(axis=0)
            top_10_indices_fk = np.argsort(-mean_abs_values_fk)[:10]
            feature_names_fk = np.array(shap_values_fk.feature_names)
            feature_names_fk = feature_names_fk[top_10_indices_fk]

            X_filtered_nf = X_filtered[X_filtered["Foreign_worker"] == "A202"]
            shap_values_nf = explainer(X_filtered_nf)
            mean_abs_values_nf = np.abs(shap_values_nf.values).mean(axis=0)
            top_10_indices_nf = np.argsort(-mean_abs_values_nf)[:10]
            feature_names_nf = np.array(shap_values_nf.feature_names)
            feature_names_nf = feature_names_nf[top_10_indices_nf]

            shap_data_fk = {
                "A201": (feature_names_fk, mean_abs_values_fk[top_10_indices_fk]), 
                "A202": (feature_names_nf, mean_abs_values_nf[top_10_indices_nf]), 
            }



            # AGE 

            X_filtered_25 = X_filtered[X_filtered["Age"] <= 25]
            shap_values_25 = explainer(X_filtered_25)
            mean_abs_values_25 = np.abs(shap_values_25.values).mean(axis=0) 
            top_10_indices_25 = np.argsort(-mean_abs_values_25)[:10]
            feature_names_25 = np.array(shap_values_25.feature_names)
            feature_names_25= feature_names_25[top_10_indices_25]

            X_filtered_35 = X_filtered[X_filtered["Age"] <= 35]
            shap_values_35 = explainer(X_filtered_35)
            mean_abs_values_35 = np.abs(shap_values_35.values).mean(axis=0)
            top_10_indices_35 = np.argsort(-mean_abs_values_35)[:10]
            feature_names_35 = np.array(shap_values_35.feature_names)
            feature_names_35 = feature_names_35[top_10_indices_35]

            X_filtered_45 = X_filtered[X_filtered["Age"] <= 45]
            shap_values_45 = explainer(X_filtered_45)
            mean_abs_values_45 = np.abs(shap_values_45.values).mean(axis=0)
            top_10_indices_45 = np.argsort(-mean_abs_values_45)[:10]
            feature_names_45 = np.array(shap_values_45.feature_names)
            feature_names_45 = feature_names_45[top_10_indices_45]

            X_filtered_55 = X_filtered[X_filtered["Age"] <= 55]
            shap_values_55 = explainer(X_filtered_55)
            mean_abs_values_55 = np.abs(shap_values_55.values).mean(axis=0)
            top_10_indices_55 = np.argsort(-mean_abs_values_55)[:10]
            feature_names_55 = np.array(shap_values_55.feature_names)
            feature_names_55 = feature_names_55[top_10_indices_55]

            X_filtered_65 = X_filtered[X_filtered["Age"] <= 65]
            shap_values_65 = explainer(X_filtered_65)
            mean_abs_values_65 = np.abs(shap_values_65.values).mean(axis=0)
            top_10_indices_65 = np.argsort(-mean_abs_values_65)[:10]
            feature_names_65 = np.array(shap_values_65.feature_names)
            feature_names_65 = feature_names_65[top_10_indices_65]
            

            X_filtered_75 = X_filtered[X_filtered["Age"] <= 75]
            shap_values_75 = explainer(X_filtered_75)
            #shap.plots.bar(shap_values_75)
            mean_abs_values_75 = np.abs(shap_values_75.values).mean(axis=0)
            top_10_indices_75 = np.argsort(-mean_abs_values_75)[:10]
            feature_names_75 = np.array(shap_values_75.feature_names)
            feature_names_75 = feature_names_75[top_10_indices_75]

            shap_data_age = {
                "19-25": (feature_names_25, mean_abs_values_25[top_10_indices_25]), 
                "26-35": (feature_names_35, mean_abs_values_35[top_10_indices_35]), 
                "36-45": (feature_names_45, mean_abs_values_45[top_10_indices_45]), 
                "46-55": (feature_names_55, mean_abs_values_55[top_10_indices_55]), 
                "56-65": (feature_names_65, mean_abs_values_65[top_10_indices_65]), 
                "66-75": (feature_names_75, mean_abs_values_75[top_10_indices_75]), 
            }



            #MALE FEMALE
            X_filtered_male = X_filtered[X_filtered["Personal_status"].isin(["A91", "A93", "A94"])]
            shap_values_male = explainer(X_filtered_male)

            X_filtered_female = X_filtered[X_filtered["Personal_status"].isin(["A92", "A95"])]
            shap_values_female = explainer(X_filtered_female)


            mean_abs_values_female = np.abs(shap_values_female.values).mean(axis=0) 
            #print(mean_abs_values)
            top_10_indices_female = np.argsort(-mean_abs_values_female)[:10]
            #top_10_indices_female = np.argsort(-mean_abs_values_female)[:2]
            #print(top_10_indices)
            feature_names_female = np.array(shap_values_female.feature_names)
            feature_names_female = feature_names_female[top_10_indices_female]

            mean_abs_values_male = np.abs(shap_values_male.values).mean(axis=0)
            top_10_indices_male = np.argsort(-mean_abs_values_male)[:10]
            #top_10_indices_male = np.argsort(-mean_abs_values_male)[:2]
            feature_names_male = np.array(shap_values_male.feature_names)
            feature_names_male = feature_names_male[top_10_indices_male]

            shap_data_MF = {
                "Female": (feature_names_female, mean_abs_values_female[top_10_indices_female]), 
                "Male": (feature_names_male, mean_abs_values_male[top_10_indices_male])
            }
            

            response = ""
            
            G_male_female = shap_graph_fairness(shap_data_MF, "Male_female")
            G_male_female.node_attr['shape'] = 'box' 
            #G_male_female.draw("shap_fair_graph.png", format="png", prog="dot")
            
            response_male_female = fairness_shap_nle(G_male_female, "Explanation")

            G_personal_status = shap_graph_fairness(shap_data_personal_status, "Personal_status")
            response_personal_status = fairness_shap_nle(G_personal_status, "Explanation")

            G_age = shap_graph_fairness(shap_data_age, "Age_group")
            response_age = fairness_shap_nle(G_age, "Explanation")

            G_fk = shap_graph_fairness(shap_data_fk, "Foreign_worker")
            response_fk = fairness_shap_nle(G_fk, "Explanation")

            response = response_male_female + response_personal_status + response_age + response_fk


            return JsonResponse({'answer': response})



        return JsonResponse({'answer': "Sorry, I didn't understand that."})
    return render(request, "model_explanation.html")