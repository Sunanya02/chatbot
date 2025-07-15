from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from collections import defaultdict
import re


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


app = Flask(__name__)
CORS(app)




database = {
    "CCSP": {
        "General Introduction": {
            "CCSP Definition": {
                "What is CCSP and why is it important?": "CCSP is C-DOT’s oneM2M-compliant IoT platform, declared a national standard by TEC. It provides a unified, secure, and scalable framework for deploying IoT applications across domains like transport, energy, and agriculture.",
                "Is CCSP specific to IR-NIYANTRAC or used elsewhere too?": "CCSP is a national platform used across multiple IoT/M2M projects. IR-NIYANTRAC is one deployment, but the same platform supports smart energy, surveillance, buildings, and more.",
                "What does it mean that CCSP supports multiple verticals?": "It means CCSP offers standardized APIs and services that work for different sectors like transport or energy — so vendors don’t need to adapt their systems to each domain separately."
            }
        },
        "CCSP Services": {
            "Common Service Functions (CSFs)": {
                "What are Common Service Functions (CSFs) in CCSP?": "CSFs are standardized modules like security, registration, device management, and data handling. They form the base layer for your device communication and operations.",
                "Do I need to implement these CSFs on my device?": "No. These functions are managed by the CCSP server. Your device just needs to interact using standard oneM2M messages and configurations as guided in onboarding.",
                "What is the benefit of having these CSFs?": "They simplify development by abstracting complexities like secure access, message routing, and device discovery. You just focus on your app logic and payloads"
            },
            "Interworking and Connectivity": {
                "Does CCSP support devices using NB-IoT or Non-IP protocols?": "Yes, CCSP supports 3GPP interworking like NIDD for NB-IoT devices using the T8 interface. You can onboard such devices through special nodes or access gateways.",
                "What is the MN or ASN in CCSP architecture?": "MN (Middle Node) and ASN (Access Service Node) are part of the CCSP structure that allow multiple devices to register and communicate securely via a shared node.",
                "What is an SSP and how does it affect my device?": "An SSP (Service Subscription Provider) manages logical groupings of vendors/devices. Your device’s configuration and access are scoped to the SSP you’re assigned during onboarding."
            },
            "Data and Device Management": {
                "What is a Flex Container and why is it used in CCSP?": "A Flex Container is a flexible data holder in the oneM2M hierarchy. It stores structured sensor data, config values, or diagnostic information, based on your payload design.",
                "How does CCSP support device management protocols?": "CCSP supports device management using LwM2M and TR-069. If your device supports either, it can be remotely configured, monitored, or updated via these standards.",
                "Who manages the device configurations on CCSP?": "Vendors are onboarded with a config file, but further device management is usually done by the SSE or CRIS team using remote management tools and CCSP’s interfaces"
            }
        },
        "CCSP Benefits": {
            "Benefits for Vendors": {
                "Why is CCSP useful for application providers like us?": "CCSP simplifies secure communication, payload handling, and device discovery. You can onboard and start pushing messages using familiar REST APIs or MQTT.",
                "Can we integrate our dashboard or system with CCSP?": "Yes. The platform provides standard oneM2M interfaces (HTTP, CoAP, MQTT), so your server or app can fetch data, subscribe to notifications, or trigger updates easily.",
                "Does CCSP ensure interoperability across different vendors?": "Yes. That’s the biggest strength of CCSP — multiple vendors and device types can work together using oneM2M APIs, no matter the hardware or network."
            }
        }
    },
    "oneM2M": {
        "OneM2M basics": {
            "OneM2M for Beginner": {
                "Where can I learn the basics of oneM2M if I’m new to it?": "You can start with the oneM2M.org introduction page, refer to C-DOT’s oneM2M guide site (https://coi.cdot.in/docs/oneM2MGuideline.html), and explore the IR-NIYANTRAC SDK — all designed to explain the key concepts clearly for new vendors.",
                "What is an AE-ID and why is it needed in the oneM2M structure?": "AE-ID is a unique identifier for your Application Entity. It helps the platform recognize your device and route messages, containers, and subscriptions correctly.",
                "What is the significance of container and contentInstance in oneM2M?": "A container holds data entries from the device as child resources in the form of CIs. Each contentInstance (CI) represents one payload/message sent by the device, stored in that container (the actual data).",
                "How do subscriptions work in oneM2M?": "When a new CI or event occurs, the subscription sends a notification to the registered nu (notification URI). It helps your device or server stay updated in real-time.",
                "What role does the App-ID registry play in oneM2M?": "App-ID registration is your first onboarding step. It uniquely identifies your vendor/application and is used to map AE-IDs and device configurations securely.",
                "Are there any tools to simulate oneM2M messages for testing?": "Yes. Tools like Postman, oneM2M Node Simulator, or in-house vendor tools (as guided by C-DOT or SSE) can simulate AE creation, CI push, and subscription flows.",
                "Can I send oneM2M-compliant messages from any device or is certification required?": "Any device can send messages if it follows the correct format and TLS setup. However, approved configurations and certificates must be issued via SSE during onboarding."
            }
        }
    },
    "IR-NIYANTRAC": {
        "Vendor Onboarding": {
            "App-ID Registration": {
                "After registering, do I need to inform anyone?": "Yes. Once you complete the App-ID registration, you must share the App-ID and relevant device/vendor details with your assigned SSE (Station Engineer). Registration alone does not trigger the vendor configuration process.",
                "I submitted the App-ID but didn’t save the confirmation. How do I retrieve it?": "If you didn’t save the confirmation screen or email, you can re-login to the App-ID portal and check your submitted entries or, contact the C-DOT onboarding support team with your registered email or organization name to retrieve the details.",
                "Should I wait for approval, or can I proceed after registering?": "You can proceed with device onboarding after App-ID registration. However, you must wait for the SSE to complete your vendor configuration before moving further.",
                "Where can I check the status of my App-ID registration?": "Currently, there may not be a live tracking portal. You can check your email inbox for confirmation or, follow up with your SSE or the registration support contact provided.",
                "I submitted my App-ID incorrectly — what should I do?": "If there's a mistake: do not submit a new App-ID again immediately. Contact the C-DOT team or your SSE and request a correction. They may allow you to re-submit or update the existing entry depending on the stage of processing.",
                "Can I reuse the same App-ID for multiple devices?": "Not specified."
            },
            "Sending Details to SSE / Vendor Configuration": {
                "Whom do I send my vendor details to after App-ID registration?": "Not specified.",
                "I sent the details to the SSE, but haven’t received a reply — what now?": "If there's no response after a reasonable wait (2–3 working days), follow up via a polite reminder email to the same thread CC-ing the official support contact or your CRIS counterpart if applicable.",
                "How long does it take for the SSE to respond or configure my IFD?": "Response times can vary. Generally initial acknowledgment may take 2–3 working days. Configuration (on COI or production) can take 1–2 weeks, depending on the complexity and queue. Always confirm timelines with your SSE.",
                "I received the LoA, but I don’t know the correct SSE contact. What should I do?": "Not specified."
            }
        },
        "Device Onboarding": {
            "Configuration File & IFD Info": {
                "I received a config file — where do I find instructions to use it?": "Instructions for using the config file are shared during onboarding via email,usually along with the file itself. If not, refer to the Device Onboarding Guide (shared by C-DOT) or check with the assigned SSE.",
                "What exactly is in the config file, and how do I interpret it?": "The config file contains: IFD identifier details, Network endpoint and broker info, Device-specific App-ID and AE-ID, Security credentials like TLS parameters. You can open it in a text editor. The file follows key-value format, often in JSON or .properties style. Refer to the onboarding guide to understand each field.",
                "How do I confirm my configuration file is the required one?": "Check the AE-ID, App-ID, and IFD name in the config file. They should match your assigned vendor information. If unsure, verify with the SSE who shared the file or your CRIS SPOC.",
                "I lost the original config file — can the SSE re-send it?": "Yes. Raise a request to the concerned SSE team or CRIS contact. Provide your IFD name and App-ID to regenerate or retrieve the config.",
                "My device model doesn’t match the config — will it cause issues?": "Yes, potentially. The config is often tailored for a specific device class or firmware. Inform the SSE or CRIS contact immediately — a mismatch may affect connectivity or payload parsing."
            },
            "Pre-Testing & COI Setup": {
                "How do I request access to the COI (test) infrastructure?": "Contact your CRIS SPOC and provide your App-ID, vendor name, and IFD model. They will forward your details to C-DOT to initiate test access setup.",
                "Who initiates the mail to C-DOT for test infra access — me or CRIS?": "CRIS initiates the mail to C-DOT on your behalf. Ensure you provide all required information to them in advance.",
                "How long after CRIS sends the mail will I get test access?": "It typically takes 3–5 working days, depending on queue and completeness of the shared info. Follow up if you don't get access after a week.",
                "Can I test multiple IFDs on the same COI setup?": "Yes, you can test multiple IFDs if all are configured correctly and uniquely identifiable (distinct AE-IDs, App-IDs, etc.). Ensure each IFD has a valid configuration and certificate.",
                "Is my vendor config on COI different from production?": "Yes. The COI (test) environment has separate endpoints and may use dummy or test certificates. While the structure is similar, do not reuse production credentials/configs in COI and vice versa."
            },
            "Certificate Generation (Clause 16)": {
                "Where do I find the steps to generate my device SSL certificate?": "The complete steps are available in the Certificate Generation SOP shared by C-DOT during onboarding. Typically, this includes instructions for generating a private key and CSR (Certificate Signing Request), filling the device identity fields (Common Name, App-ID, etc.) and submitting the CSR to the designated C-DOT portal or SSE. If you haven’t received the SOP, contact your SSE or CRIS representative.",
                "I wrongly filled the certificate fields — can I regenerate it?": "Yes. You can safely regenerate a new CSR using the correct values and resubmit it. Ensure that the Common Name (CN), App-ID, and other fields match the config exactly. If a signed certificate has already been issued, notify your SSE to revoke the old one.",
                "Do I need to use a specific tool to generate the CSR?": "You can use standard tools like openssl (Linux/Windows) and easyRSA (Linux/Windows). C-DOT recommends easyRSA for most vendors. The SOP includes exact command examples.",
                "My certificate file is in .pem — is that okay?": "Yes, .pem format is acceptable and commonly used. Just ensure that the file contains the full certificate chain (if required) and matches the format expected by your device and the CCSP broker.",
                "Where do I upload the certificate after generating?": "You do not upload it yourself. Instead send the CSR to your SSE or the certificate issuance portal. Once signed, the SSE or test coordinator will provide the signed certificate. You must then install it on your IFD manually or via config scripts.",
                "My certificate got signed but it’s not working — what should I check?": "Ensure you’re using the correct private key that was used to generate the CSR. Check the certificate format and ensure it’s not corrupted. Verify the Common Name (CN) and App-ID fields match what your broker expects. Make sure the certificate is installed in the correct path on your device. Use tools like openssl s_client or device debug logs to inspect handshake failures.",
                "What’s the difference between root CA and device certificate?": "A Root CA is a trusted entity that signs other certificates. It must be pre-installed on your device and Device Certificate is issued specifically to your IFD and proves its identity during TLS handshake.",
                "How do I install the certificate on my IFD?": "The method depends on your device platform. Typically copy the .pem certificate and private key into the correct location (e.g., /etc/certs/). Update your MQTT/HTTP client to reference those files. Restart the client or service. For embedded systems, this may be done via firmware scripts or manually over a debug console."
            },
            "Connectivity & Message Errors": {
                "Which documentation should I refer to for connecting to the broker?": "Connection details (broker URL, port, TLS settings) are shared in the device onboarding email or config file by your SSE. Please refer to that file and the Device Onboarding Guide.",
                "I got a TLS handshake failure — what could be wrong?": "This usually means the certificate in your config file is missing or mismatched. Recheck the TLS settings, and if needed, contact your SSE or CRIS SPOC to regenerate the certificate.",
                "My MQTT client gives 'connection refused' — what should I check?": "Ensure you are using the correct broker IP and port from the config file, and that the certificate is correctly loaded. If the issue continues, request your SSE to verify backend readiness.",
                "I connected but no data is going through — what can I debug?": "Make sure you’ve created the correct container and subscription under your AE. Also check if your AE-ID, App-ID, and topic structure match the onboarding config. SSE can help verify.",
                "My device shows 'broker not reachable' — what can I do?": "Check internet access and broker IP in the config. If using COI, confirm the test infra is set up. If the broker is still unreachable, reach out to your SSE or CRIS contact for support."
            },
            "Understanding resource hierarchy": {
                "Container and Subscription Creation": {
                    "How do I know which containers I must create and which are already created?": "Refer to the onboarding instructions or sample AE structure shared by your SSE. You can also view the existing containers under your AE in the CCSP portal’s resource tree.",
                    "Are there any orders in which I should create the containers and subscriptions - where should I refer for them?": "Yes, always create AE first, then containers, then subscriptions. This order is important. Follow the creation steps in the onboarding email or the oneM2M Container Guide (Clause 8).",
                    "I created a container but it is not visible in the resource hierarchy — what’s wrong?": "The creation may have failed or happened under a different AE. Check the API response or try refreshing the CCSP portal. If unsure, share your AE-ID and container path with your SSE.",
                    "Where can I verify the container/subscription creation on the CCSP portal?": "Login to the CCSP portal and open the Resource Tree. Expand your AE to view all child containers and subscriptions created under it.",
                    "The subscription was accepted but no notifications are coming — whom should I contact?": "Check if your listener is running and the nu (notification URI) is correct. If everything looks fine, contact your SSE or backend SPOC to check notification delivery.",
                    "Can I delete a container and recreate it?": "Yes, you can delete and recreate it. But note that this will remove all data and subscriptions under the original container, so proceed carefully.",
                    "While viewing the resourceTree it shows “Unexpected Error Occured, Contact Administrator” - what should I do now?": "Try logging out and back in or refreshing the page. If the error persists, share the timestamp and AE-ID with the admin or backend SPOC for investigation."
                }
            },
            "Understanding message flow": {
                "Payload and Communication Format": {
                    "Where can I find example payloads?": "Go to the 'IR-NIYANTRAC Messages' section in the SDK. It includes sample payloads, data type examples, and correct field structures as per Clause 10 — all mapped to oneM2M standards for easier reference.",
                    "My payload structure gives '400 Bad request' — which guide should I refer to?": "Check the payload format against the Message Communication Format in the SDK (Messages section). Common issues include missing fields or invalid data types.",
                    "Should the timestamps in payloads be in UTC?": "Yes. Timestamps like ct, lt, and et values must be in UTC ISO 8601 format (e.g., 2025-06-19T09:00:00).",
                    "I sent a CI but it’s not reflected on the resourceTreeViewer — how do I debug it?": "First confirm the container exists. Then verify CI response (should be 201 Created). If all looks fine, wait a few seconds and refresh. Still stuck? Contact your SSE.",
                    "I am having a hard time creating payloads with values - what should I refer to better understand the attributes in payload?": "Use the 'Messages' section in the SDK. It explains oneM2M fields, expected data types, and sample values for each key — no need to refer to external guides."
                }
            }
        }
    }
}



def preprocess_text(text, for_keyword=False):
   
    text = text.lower()
    
    phrases = ['vendor onboarding', 'app-id registration', 'device onboarding', 'sse vendor configuration', 'ccsp', 'onem2m', 'ir-niyantrac']
    for phrase in phrases:
        text = text.replace(phrase, phrase.replace(' ', '_'))
   
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in string.punctuation]
   
    if not for_keyword:
        
        stop_words = set(['is', 'are', 'the', 'a', 'an', 'and', 'or', 'but'])
        tokens = [t for t in tokens if t not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
   
    return ' '.join(tokens)



category_keywords = {
    'CCSP': ['ccsp', 'iot platform', 'onem2m'],
    'oneM2M': ['onem2m', 'm2m', 'ae-id', 'container'],
    'IR-NIYANTRAC': ['ir-niyantrac', 'niyantrac'],
    'Vendor Onboarding': ['vendor onboarding', 'vendor registration', 'app-id', 'sse configuration'],
    'Device Onboarding': ['device onboarding', 'config file', 'ifd', 'coi setup', 'certificate generation'],
    'Understanding resource hierarchy': ['resource hierarchy', 'container creation', 'subscription creation'],
    'Understanding message flow': ['message flow', 'payload format', 'communication format']
}


def get_category_score(query, category, subcategory=None):
    query = preprocess_text(query, for_keyword=True)
    keywords = category_keywords.get(category if subcategory is None else subcategory, [])
    score = 0
    for keyword in keywords:
        if keyword.replace('_', ' ') in query:
            score += 0.5
        elif any(word in query for word in keyword.split()):
            score += 0.1  
    return score



all_questions = []
question_paths = []
category_map = defaultdict(list)


def collect_questions(db, path=[]):
    for key, value in db.items():
        new_path = path + [key]
        if isinstance(value, dict):
            collect_questions(value, new_path)
        else:
            all_questions.append(value)
            question_paths.append(new_path)
            
            if len(new_path) >= 1:
                category_map[new_path[0]].append(len(all_questions) - 1)
                if len(new_path) >= 2:
                    category_map[new_path[1]].append(len(all_questions) - 1)


collect_questions(database)
processed_questions = [preprocess_text(q) for q in all_questions]


@app.route('/')
def serve_index():
    print(f"Serving static file: {os.path.join(app.static_folder, 'index.html')}")
    if os.path.exists(os.path.join(app.static_folder, 'index.html')):
        return send_from_directory('static', 'index.html')
    else:
        return jsonify({'error': 'index.html not found in static folder'}), 404


@app.route('/get_answer', methods=['POST'])
def get_answer():
    data = request.json
    path = data.get('path', [])
    if not path or len(path) < 4:
        return jsonify({'error': 'Invalid path'}), 400
    try:
        answer = database
        for key in path:
            answer = answer[key]
        return jsonify({'answer': answer})
    except KeyError:
        return jsonify({'error': 'Question not found'}), 404
    
    
@app.route("/api")
def api():
    return "Hello, API"


@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.json
    user_question = data.get('question', '')
    if not user_question:
        return jsonify({'error': 'No question provided'}), 400


    processed_user_question = preprocess_text(user_question)
   
    
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(processed_questions + [processed_user_question])
    similarities = cosine_similarity(question_vectors[-1], question_vectors[:-1])
   
  
    category_scores = {}
    for category in category_keywords:
        score = get_category_score(user_question, category)
        if score > 0:
            category_scores[category] = score
   
    
    combined_scores = similarities[0].copy()
    for category, score in category_scores.items():
        for idx in category_map.get(category, []):
            combined_scores[idx] += score
   
    max_score_idx = combined_scores.argmax()
    max_score = combined_scores[max_score_idx]
   
    if max_score > 0.15:  
        answer = all_questions[max_score_idx]
    else:
        answer = "Sorry, I couldn't find a precise answer. For Vendor Onboarding, please check the 'IR-NIYANTRAC' > 'Vendor Onboarding' section for details on App-ID Registration and SSE configuration."
   
    return jsonify({'answer': answer})


if __name__ == '__main__':
    app.run(debug=True)
