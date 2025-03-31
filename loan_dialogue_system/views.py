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

MODEL_PATH = os.path.join(os.path.dirname(__file__), "catboost_model_wrapped.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)



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
            model_prediction = model.predict(input_df)
            # print("Prediction made")  # Debugging step
            # print(prediction)
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
            model_exp = dice_ml.Model(model=model, backend="sklearn", model_type='classifier')
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
            
            nle = generate_nle(G, "Explanation")
            # print(nle)

            #print(identify_changes(exp_df.iloc[0], input_df))
            request.session['prediction'] = prediction
            request.session['loan_data'] = input_data
            context = {
                'prediction': prediction,
                'nle': nle,
                
            }

            return render(request, "check_loan.html", context)

            
        except Exception as e:
            print(e)
            return HttpResponse("Invalid input")
    



    return render(request, "check_loan.html")

def loan_prediction(request):
    return render(request, "loan_prediction.html")
 




domain_knowledge = {
    'Account_status':
    {
        "description": "Status of existing checking account",
        "values": {
            "A11": "< 0 DM",
            "A12": "0 <= ... < 200 DM",
            "A13": ">= 200 DM",
            "A14": "no checking account",
        }
    },
    'Months':
    {
        "description": "Duration in months",
    },
    'Credit_history':
    {
        "description": "Credit history",
        "values": {
            "A30": "no credits taken/ all credits paid back duly",
            "A31": "all credits at this bank paid back duly",
            "A32": "existing credits paid back duly till now",
            "A33": "delay in paying off in the past",
            "A34": "critical account/ other credits existing (not at this bank)",
        }
    },
    'Purpose':
    {
        "description": "Purpose",
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
    'Credit_amount':
    {
        "description": "Credit amount",
    },
    'Savings':
    {
        "description": "Savings account/bonds",
        "values": {
            "A61": "< 100 DM",
            "A62": "100 <= ... < 500 DM",
            "A63": "500 <= ... < 1000 DM",
            "A64": ">= 1000 DM",
            "A65": "unknown/ no savings account",
        }
    },
    'Employment':
    {
        "description": "Present employment since",
        "values": {
            "A71": "unemployed",
            "A72": "< 1 year",
            "A73": "1 <= ... < 4 years",
            "A74": "4 <= ... < 7 years",
            "A75": ">= 7 years",
        }
    },
    'Installment_rate':
    {
        "description": "Installment rate in percentage of disposable income",
    },
    'Personal_status':
    {
        "description":"Personal status and sex",
        "values": {
            	"A91" : "male: divorced/separated",
                "A92" : "female: divorced/separated/married",
                "A93" : "male: single",
                "A94": "male: married/widowed",
                "A95":"female: single",
        }

    },
    "Other_debtors":
    {
        "description": "Other debtors/ guarantors",
        "values": {
            "A101": "none",
            "A102": "co-applicant",
            "A103": "guarantor",
        }
    },
    "Residence":
    {
        "description": "Present residence since",
    },
    "Property":
    {
        "description": "Property",
        "values": {
            "A121": "real estate",
            "A122": "if not real estate: building society savings agreement/ life insurance",
            "A123": "if not real estate/building society savings agreement/ life insurance:  car or other, not in Savings account/bonds",
            "A124": "unknown / no property",
        }
    },
    "Age":
    {
        "description": "Age in years",
    },
    "Other_installments":
    {
        "description": "Other installments",
        "values": {
            "A141": "bank",
            "A142": "stores",
            "A143": "none",
        }
    },
    "Housing":
    {
        "description": "Housing",
        "values": {
            "A151": "rent",
            "A152": "own",
            "A153": "for free",
        }
    },
    "Number_credits":
    {
        "description": "Number of existing credits at this bank",
    },
    "Job":
    {
        "description": "Job",
        "values": {
            "A171": "unemployed/ unskilled - non-resident",
            "A172": "unskilled - resident",
            "A173": "skilled employee / official",
            "A174": "management/ self-employed/ highly qualified employee/ officer",
        }
    },
    "Number_dependents":
    {
        "description": "Number of people being liable to provide maintenance for",
    },
    "Telephone":
    {
        "description": "Telephone",
        "values": {
            "A191": "none",
            "A192": "yes",
        }
    },
    "Foreign_worker":
    {
        "description": "Foreign worker",
        "values": {
            "A201": "yes",
            "A202": "no",
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

def create_explanation_graph(counterfactual_instance, prediction, input_df):
    changes = identify_changes(counterfactual_instance, input_df)
    G = nx.DiGraph()
    G.add_node("Explanation", description="Explanation of model prediction changes")
    G.add_node(prediction, description=f"Model predicted '{prediction}'")
    G.add_edge("Explanation", prediction, relation="ModelOutput")

    for feature, (original_value, new_value) in changes.items():
        G.add_node(f"{feature}", description = f"{feature}")
        feature_descr = domain_knowledge[feature]["description"]
        G.add_node(f"{feature}: {feature_descr}", description=feature_descr)
        G.add_edge(f"{feature}: {feature_descr}", f"{feature}", relation="Describes")
        G.add_node(f"{feature}: {original_value}", description=original_value)
        G.add_node(f"{feature}: {new_value}", description=new_value)
        G.add_edge("Explanation", f"{feature}", relation="Changed")
        G.add_edge(f"{feature}", f"{feature}: {original_value}", relation="Original Value")
        G.add_edge(f"{feature}", f"{feature}: {new_value}", relation="New Value")
        G.add_edge(f"{feature}: {original_value}", f"{feature}: {new_value}", relation="Changed To")
        if "values" in domain_knowledge[feature]:
            
            feature_val_original = domain_knowledge[feature]["values"][original_value]
            G.add_node(f"{feature}: {feature_val_original}", description=feature_val_original)
            G.add_edge(f"{feature}: {original_value}", f"{feature}: {feature_val_original}", relation="Describes")

            feature_val_new = domain_knowledge[feature]["values"][new_value]
            G.add_node(f"{feature}: {feature_val_new}", description=feature_val_new)
            G.add_edge(f"{feature}: {new_value}", f"{feature}: {feature_val_new}", relation="Describes")

            G.add_edge(f"{feature}: {feature_val_original}", f"{feature}: {feature_val_new}", relation="Changed To")

    
    return G

def depth_first_search(graph, node, visited, explanation):
    visited.add(node)
    for neighbor in graph.neighbors(node):
        relation = graph.get_edge_data(node, neighbor)["relation"]
        if relation == "ModelOutput":
            explanation.insert(0, f"The prediction of '{neighbor}' is based on the following factors:")
        if relation == "Changed":
            explanation.append(f"The {node} has changed to {neighbor}")
        if relation == "Changed To":
            explanation.append(f"The {node} has changed to {neighbor}")
        if neighbor not in visited:
            depth_first_search(graph, neighbor, visited, explanation)
    return explanation

def generate_nle(graph, root_node):
    explanation = []
    explanation = depth_first_search(graph, root_node, set(), explanation)
    print(explanation)
    
    return "\n".join(explanation)

def generate_counterfactual(input_data, prediction):
    print("DEBUG")
    input_df = pd.DataFrame(input_data, index=[0])
    print("dataframe")
    print(input_df)
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
    model_exp = dice_ml.Model(model=model, backend="sklearn", model_type='classifier')
    exp = dice_ml.Dice(data, model_exp, method="random")
    exp_user = exp.generate_counterfactuals(input_df, total_CFs=2, desired_class="opposite")
    # exp_user.visualize_as_dataframe(show_only_changes=True)
    exp_df = exp_user.cf_examples_list[0].final_cfs_df
    # print(exp_df)
    G0 = create_explanation_graph(exp_df.iloc[0], prediction, input_df)
   
    G1 = create_explanation_graph(exp_df.iloc[1], prediction, input_df)
    nle0 = generate_nle(G0, "Explanation")
    nle1 = generate_nle(G1, "Explanation")
    nle = nle0 + "<br>" + nle1
    return nle


def generate_feature_importance(input_data):
    explainer = shap.Explainer(model)
    input_df = pd.DataFrame(input_data, index=[0])
    shap_values = explainer(input_df)
    
    #shap.plots.waterfall(shap_values[0])
    #shap.force_plot(shap_values[0], input_df.iloc[0,:])
    #shap.force_plot(explainer.expected_value, shap_values.values[0, :], input_df.iloc[0,:])
    return shap_values
'''
def shap_graph(features, data, feature_contributions, prediction):
    G = nx.DiGraph()
    G.add_node("Explanation", description="Explanation of model prediction changes")

    G.add_node("Negative contributions", description="Negative contributions")
    G.add_node("Positive contributions", description="Positive contributions")
    G.add_edge("Explanation", "Negative contributions", relation="ModelOutput")
    G.add_edge("Explanation", "Positive contributions", relation="ModelOutput")

    for i in range(3):
        G.add_node(features[i], description=features[i])
        if feature_contributions[i] < 0:
            G.add_edge("Negative contributions", features[i], relation="Contributes")
        else:
            G.add_edge("Positive contributions", features[i], relation="Contributes")
        feature_descr = domain_knowledge[features[i]]["description"]

        G.add_node(feature_descr, description=feature_descr)
        G.add_edge(features[i], feature_descr, relation="Describes")

        G.add_node(feature_contributions[i], description=feature_contributions[i])
        G.add_edge(feature_contributions[i], features[i], relation="Contributes")

        G.add_node(data[i], description=data[i])
        G.add_edge(data[i], features[i], relation="Value of feature")

        if "values" in domain_knowledge[features[i]]:
            feature_val = domain_knowledge[features[i]]["values"][data[i]]
            G.add_node(feature_val, description=feature_val)
            G.add_edge(data[i], feature_val, relation="Describes")
            #G.add_node(f"{features[i]}: {feature_contributions[i]}", description=features[i])
            #G.add_edge("Negative contributions", f"{features[i]}: {feature_contributions[i]}", relation="Contributes")


    return G'
'''

def shap_graph(features, data, feature_contributions, prediction):
    G = nx.DiGraph()
    G.add_node("Explanation", description="Explanation of model prediction changes")
    G.add_node(prediction, description=f"Model predicted '{prediction}'")
    G.add_edge("Explanation", prediction, relation="ModelOutput")

    for i in range(len(data)):
        G.add_node(data[i], description=data[i])
        if feature_contributions[i] < 0:
            G.add_edge("Explanation", data[i], relation="Negative Contribution")
            
        else:
            G.add_edge("Explanation", data[i], relation="Positive Contribution")
        
        G.add_node(features[i], description=features[i])
        G.add_edge(data[i], features[i], relation="Value of feature")
        feature_descr = domain_knowledge[features[i]]["description"]
        G.add_node(feature_descr, description=feature_descr)
        G.add_edge(features[i], feature_descr, relation="Describes")
        if "values" in domain_knowledge[features[i]]:
            data_val = domain_knowledge[features[i]]["values"][data[i]]
            G.add_node(data_val, description=data_val)
            G.add_edge(data[i], data_val, relation="Describes")
        if feature_contributions[i] < 0:
            G.add_node("Negative contribution", description="Negative contribution")
            G.add_edge(data[i], "Negative contribution", relation="Contributes")
    return G


def depth_first_search_shap(graph, node, visited, explanation):
    visited.add(node)
    for neighbor in graph.neighbors(node):
        relation = graph.get_edge_data(node, neighbor)["relation"]
        if relation == "ModelOutput":
            explanation.insert(0, f"The prediction of '{neighbor}' is based on the following factors:")
        
        if relation == "Describes":
            explanation.append(f"The {node} describes the feature")
        if relation == "Value of feature":
            explanation.append(f"The value of the feature is {node}")
        if relation == "Negative Contribution":
            explanation.append(f"The {node} has a negative contribution")
        if relation == "Positive Contribution":
            explanation.append(f"The {node} has a positive contribution")
       
        if neighbor not in visited:
            depth_first_search(graph, neighbor, visited, explanation)
    return explanation


def shap_nle(graph, root_node):
    explanation = []
    explanation = depth_first_search_shap(graph, root_node, set(), explanation)
    print(explanation)
    
    return "\n".join(explanation)


def chatbot(request):
    if request.method == "POST":
        prediction = request.session.get("prediction")
        loan_data = request.session.get("loan_data")
        
        if not prediction or not loan_data:
            return JsonResponse({'error': 'Please submit a loan application form.'})
        

        user_input = request.POST.get("user_input")


        if user_input == "Remind me my loan status?":
            response = prediction
            return JsonResponse({'answer': response})
        
        if user_input == "How would my loan status be different?":
            response = generate_counterfactual(loan_data, prediction)
            return JsonResponse({'answer': response})
        if user_input == "What features contributed to the prediction?":
            shap_values = generate_feature_importance(loan_data)
            feature_contributions = shap_values.values[0, :]
            feature_names = shap_values.feature_names
            data = shap_values.data[0, :]
            # print(feature_contributions)
            # print(feature_names)
            # print(data)
           
            # print(f'size of data: {data.shape} and size of feature contributions : {feature_contributions.shape}')
            # G = shap_graph(feature_names, data, feature_contributions, prediction)
            # pos = nx.planar_layout(G)
            # nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10)
            # edge_labels = nx.get_edge_attributes(G, 'relation')
            # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            # plt.show()
            G_shap = shap_graph(feature_names, data, feature_contributions, prediction)
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
            explainer = shap.Explainer(model)
            
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

            

            y_pred = model.predict(X_test)
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
            explainer = shap.Explainer(model)
            
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


            interactions = model.get_feature_importance(data_pool, type = "Interaction")
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
            

            feature_importances = model.get_feature_importance()
            print(pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False))
            print(X["Foreign_worker"].value_counts())
            return JsonResponse({'answer': "Sorry, I didn't understand that."})


        return JsonResponse({'answer': "Sorry, I didn't understand that."})
    return render(request, "other_chatbot.html")