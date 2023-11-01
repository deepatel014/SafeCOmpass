from flask import Flask, render_template, request
import pickle
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
# with open('svm_model.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)
model_paths = {
    'Logistic Regression': 'log_grid_model.pkl',
    'SVM': 'svm_model_money.pkl',
    'Neural Networks':'NeuralNetwork_grid_model.pkl',
    'Random Forest Classifier':'RandomForest_grid_model.pkl',
    'Decision Tree':'DecisionTree_grid_model.pkl'
}


# Load the preprocessing pipeline
with open('full_pipeline_money (1).pkl', 'rb') as pipeline_file:
    preprocessing_pipeline = joblib.load(pipeline_file)

selections = {}
selections['VISIBILITY'] = ['Clear', 'Drifting Snow', 'Fog, Mist, Smoke, Dust',
    'Freezing Rain',  'Other',  'Rain',  'Snow',  'Strong wind']
selections['RDSFCOND'] = ['Dry', 'Wet' ,'Other', 'Slush' ,'Loose Snow' ,'Ice', 'Packed Snow',
 'Spilled liquid' ,'Loose Sand or Gravel']

selections['SEASON'] = ['Spring','Summer','Autumn','Winter']

selections['DISTRICT'] = ['Toronto and East York', 'Scarborough' ,'Etobicoke York' ,'North York',
 'Toronto East York']
selections['LOCCOORD'] = ['Intersection' ,'Mid-Block', 'Exit Ramp Westbound',
 'Exit Ramp Southbound' ,'Mid-Block (Abnormal)', 'Entrance Ramp Westbound',
 'Park, Private Property, Public Lane']
selections['lights'] = ['Dark', 'Dark, artificial', 'Dawn', 'Dawn, artificial',
    'Daylight', 'Daylight, artificial', 'Dusk', 'Dusk, artificial', 'Other']
neighborhood_options  = {
    
    '129':"Agincourt North",
    '128':"Agincourt South-Malvern West",
    '20':"Alderwood",
    '95':"Annex",
    '153':"Avondale",
    '42':"Banbury-Don Mills",
    '34':"Bathurst Manor",
    '169':"Bay-Cloverhill",
    '52':"Bayview Village",
    '49':"Bayview Woods-Steeles",
    '39':"Bedford Park-Nortown",
    '112':"Beechborough-Greenbrook",
    '157':"Bendale South",
    '156':"Bendale-Glen Andrew",
    '122':"Birchcliffe-Cliffside",
    '24':"Black Creek",
    '69':"Blake-Jones",
    '108':"Briar Hill-Belgravia",
    '41':"Bridle Path-Sunnybrook-York Mills",
    '57':"Broadview North",
    '30':"Brookhaven-Amesbury",
    '71':"Cabbagetown-South St.James Town",
    '109':"Caledonia-Fairbank",
    '96':"Casa Loma",
    '133':"Centennial Scarborough",
    '167':"Church-Wellesley",
    '120':"Clairlea-Birchmount",
    '33':"Clanton Park",
    '123':"Cliffcrest",
    '92':"Corso Italia-Davenport",
    '66':"Danforth",
    '59':"Danforth East York",
    '47':"Don Valley Village",
    '126':"Dorset Park",
    '172':"Dovercourt Village",
    '155':"Downsview",
    '168':"Downtown Yonge East",
    '83':"Dufferin Grove",
    '62':"East End-Danforth",
    '148':"East L'Amoreaux",
    '152':"East Willowdale",
    '9':"Edenbridge-Humber Valley",
    '138':"Eglinton East",
    '5':"Elms-Old Rexdale",
    '32':"Englemount-Lawrence",
    '11':"Eringate-Centennial-West Deane",
    '159':"Etobicoke City Centre",
    '13':"Etobicoke West Mall",
    '150':"Fenside-Parkwoods",
    '44':"Flemingdon Park",
    '102':"Forest Hill North",
    '101':"Forest Hill South",
    '163':"Fort York-Liberty Village",
    '25':"Glenfield-Jane Heights",
    '141':"Golfdale-Cedarbrae-Woburn",
    '65':"Greenwood-Coxwell",
    '140':"Guildwood",
    '165':"Harbourfront-CityPlace",
    '53':"Henry Farm",
    '88':"High Park North",
    '87':"High Park-Swansea",
    '134':"Highland Creek",
    '48':"Hillcrest Village",
    '161':"Humber Bay Shores",
    '8':"Humber Heights-Westmount",
    '21':"Humber Summit",
    '22':"Humbermede",
    '106':"Humewood-Cedarvale",
    '125':"Ionview",
    '158':"Islington",
    '90':"Junction Area",
    '171':"Junction-Wallace Emerson",
    '110':"Keelesdale-Eglinton West",
    '124':"Kennedy Park",
    '78':"Kensington-Chinatown",
    '6':"Kingsview Village-The Westway",
    '15':"Kingsway South",
    '147':"L'Amoreaux West",
    '114':"Lambton Baby Point",
    '38':"Lansing-Westgate",
    '105':"Lawrence Park North",
    '103':"Lawrence Park South",
    '56':"Leaside-Bennington",
    '84':"Little Portugal",
    '19':"Long Branch",
    '146':"Malvern East",
    '145':"Malvern West",
    '29':"Maple Leaf",
    '12':"Markland Wood",
    '130':"Milliken",
    '160':"Mimico-Queensway",
    '135':"Morningside",
    '144':"Morningside Heights",
    '73':"Moss Park",
    '115':"Mount Dennis",
    '2':"Mount Olive-Silverstone-Jamestown",
    '99':"Mount Pleasant East",
    '18':"New Toronto",
    '50':"Newtonbrook East",
    '36':"Newtonbrook West",
    '68':"North Riverdale",
    '74':"North St.James Town",
    '173':"North Toronto",
    '54':"O'Connor-Parkview",
    '154':"Oakdale-Beverley Heights",
    '121':"Oakridge",
    '107':"Oakwood Village",
    '58':"Old East York",
    '80':"Palmerston-Little Italy",
    '149':"Parkwoods-O'Connor Hills",
    '23':"Pelmo Park-Humberlea",
    '67':"Playter Estates-Danforth",
    '46':"Pleasant View",
    '10':"Princess-Rosethorn",
    '72':"Regent Park",
    '4':"Rexdale-Kipling",
    '111':"Rockcliffe-Smythe",
    '86':"Roncesvalles",
    '98':"Rosedale-Moore Park",
    '89':"Runnymede-Bloor West Village",
    '28':"Rustic",
    '139':"Scarborough Village",
    '174':"South Eglinton-Davisville",
    '85':"South Parkdale",
    '70':"South Riverdale",
    '166':"St Lawrence-East Bayfront-The Islands",
    '40':"St.Andrew-Windfields",
    '116':"Steeles",
    '16':"Stonegate-Queensway",
    '118':"Tam O'Shanter-Sullivan",
    '61':"Taylor-Massey",
    '63':"The Beaches",
    '3':"Thistletown-Beaumond Heights",
    '55':"Thorncliffe Park",
    '81':"Trinity-Bellwoods",
    '79':"University",
    '43':"Victoria Village",
    '164':"Wellington Place",
    '136':"West Hill",
    '1':"West Humber-Clairville",
    '162':"West Queen West",
    '143':"West Rouge",
    '35':"Westminster-Branson",
    '113':"Weston",
    '91':"Weston-Pelham Park",
    '119':"Wexford/Maryvale",
    '37':"Willowdale West",
    '7':"Willowridge-Martingrove-Richview",
    '142':"Woburn North",
    '64':"Woodbine Corridor",
    '60':"Woodbine-Lumsden",
    '94':"Wychwood",
    '170':"Yonge-Bay Corridor",
    '151':"Yonge-Doris",
    '100':"Yonge-Eglinton",
    '97':"Yonge-St.Clair",
    '27':"York University Heights",
    '31':"Yorkdale-Glen Park"

}
@app.route('/riskmeter', methods=['GET', 'POST'])
def riskmeter():
    predicted_label = None
    if request.method == 'POST':
        form_data = request.form
        selected_model = form_data['model']  # Get the selected model name

        # Load the selected model from the corresponding pickle file
        model_path = model_paths[selected_model]
        with open(model_path, 'rb') as model_file:
            model = joblib.load(model_file)
        sample_input = {
            'AG_DRIV': int(form_data['AG_DRIV']),
            'ALCOHOL': int(form_data['ALCOHOL']),
            'AUTOMOBILE': int(form_data['AUTOMOBILE']),
            'CYCLIST': int(form_data['CYCLIST']),
            'DISTRICT': form_data['DISTRICT'],
            'HOOD_158': int(form_data['HOOD_158']),
            'LIGHT': form_data['LIGHT'],
            'LOCCOORD': form_data['LOCCOORD'],
            'PEDESTRIAN': int(form_data['PEDESTRIAN']),
            'RDSFCOND': form_data['RDSFCOND'],
            'SEASON': form_data['SEASON'],
            'SPEEDING': int(form_data['SPEEDING']),
            'TRSN_CITY_VEH': int(form_data['TRSN_CITY_VEH']),
            'TRUCK': int(form_data['TRUCK']),
            'VISIBILITY': form_data['VISIBILITY']
        }

        # Create a DataFrame from the sample input
        sample_input_df = pd.DataFrame([sample_input])

        # Transform the input using the preprocessing pipeline
        sample_input_transformed = preprocessing_pipeline.transform(sample_input_df)

        # Predict using the model
        predicted_class = model.predict(sample_input_transformed)

        # Convert the predicted class into a label
        predicted_label = 'Fatal' if predicted_class[0] == 1 else 'Non-Fatal'
        print(form_data)

        return render_template('riskmeter.html', form_data=form_data, predicted_label=predicted_label,
                               neighborhood_options=neighborhood_options, selections=selections, model_paths=model_paths)


    return render_template('riskmeter.html', form_data=None, predicted_label=None,
                           neighborhood_options=neighborhood_options, selections=selections, model_paths=model_paths)

@app.route('/')  # Add this route for the riskmeter page
def index():
    return render_template('index.html', selections=selections, neighborhood_options=neighborhood_options, model_paths=model_paths)

@app.route('/graphs')  # Add this route for the riskmeter page
def graphs():
    return render_template('graphs.html', selections=selections, neighborhood_options=neighborhood_options, model_paths=model_paths)

@app.route('/squad')  # Add this route for the riskmeter page
def squad():
    return render_template('squad.html', selections=selections, neighborhood_options=neighborhood_options, model_paths=model_paths)

@app.route('/map')  # Add this route for the riskmeter page
def map():
    return render_template('map.html', selections=selections, neighborhood_options=neighborhood_options, model_paths=model_paths)

if __name__ == '__main__':
    app.run(debug=True)
