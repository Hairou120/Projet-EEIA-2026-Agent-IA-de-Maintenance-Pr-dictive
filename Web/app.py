import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Agent IA Maintenance", page_icon="⚙️", layout="wide")

st.title("⚙️ Agent IA de Maintenance Prédictive")

tab1, tab2 = st.tabs(["📊 Diagnostic Capteurs (ML)", "💬 Assistant Manuel (RAG)"])

# ---------------------------------------------------------
# ONGLET 1 : MODULE MACHINE LEARNING
# ---------------------------------------------------------
with tab1:
    st.header("Analyse en Temps Réel des Capteurs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        air_temp = st.number_input("Température Air (K)", value=300.0, step=0.5)
        proc_temp = st.number_input("Température Procédé (K)", value=310.0, step=0.5)
        speed = st.number_input("Vitesse de Rotation (RPM)", value=1500, step=50)
    
    with col2:
        torque = st.number_input("Couple (Nm)", value=40.0, step=1.0)
        tool_wear = st.number_input("Usure Outil (min)", value=120, step=5)
    
    if st.button("🔍 Analyser l'État de la Machine", type="primary"):
        payload = {
            "air_temperature": air_temp,
            "process_temperature": proc_temp,
            "rotational_speed": speed,
            "torque": torque,
            "tool_wear": tool_wear
        }
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            res_data = response.json()
            
            if res_data["prediction"] == 1:
                st.error(f"⚠️ **ATTENTION : {res_data['status']}** (Probabilité : {res_data['failure_probability']}%)")
                
                # Couplage Smart : Proposition d'action automatique RAG
                st.info("💡 *Conseil : Allez dans l'onglet Assistant RAG et posez des questions sur la résolution des surchauffes ou l'usure d'outil.*")
            else:
                st.success(f"✅ **{res_data['status']}** (Probabilité de risque : {res_data['failure_probability']}%)")
        except Exception as e:
            st.error(f"Erreur de connexion à l'API : {e}")

# ---------------------------------------------------------
# ONGLET 2 : MODULE RAG
# ---------------------------------------------------------
with tab2:
    st.header("Assistant Technique - Manuels d'Entretien")
    
    # Historique de discussion
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_query = st.chat_input("Posez votre question sur le manuel de la machine...")
    
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Recherche dans les manuels PDF..."):
                try:
                    res = requests.post(f"{API_URL}/chat", json={"question": user_query})
                    data = res.json()
                    
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    
                    st.write(answer)
                    
                    if sources:
                        st.markdown("**📌 Sources :**")
                        for src in sources:
                            st.caption(f"- 📄 `{src['file']}` (Page {src['page']})")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Erreur : {e}")