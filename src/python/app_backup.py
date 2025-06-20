# IMPORTANT: This app requires a `requirements.txt` file in the same directory
# with the following content:
# streamlit
# pandas
# matplotlib
# seaborn
# numpy
# openpyxl

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="🎓 Inventory Management Analysis",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper function for CSV download ---
@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for downloading."""
    return df.to_csv(index=False).encode('utf-8')

# --- All Functions from the Original Script ---
# Chapter 1
def load_and_inspect_data(uploaded_file):
    """
    Loads data from a CSV or XLSX file into a pandas DataFrame.
    Automatically detects the file type.
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("❌ Unsupported file format. Please upload a CSV or XLSX file.")
            return None
        st.success(f"✅ Data successfully loaded from: '{uploaded_file.name}'")
        st.info("🔍 First 5 rows of the raw data:")
        st.dataframe(df.head())
        st.info("📋 Available columns in the dataset:")
        st.json(df.columns.tolist())
        return df
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during file loading: {e}")
        return None

# Chapter 2
def preprocess_and_aggregate_demand(df, day_col, demand_col):
    if day_col not in df.columns or demand_col not in df.columns:
        st.error(f"❌ Error: One or more required columns ('{day_col}', '{demand_col}') not found.")
        return None
    processed_df = df.copy()
    processed_df[demand_col] = pd.to_numeric(processed_df[demand_col], errors='coerce')
    initial_rows = len(processed_df)
    processed_df.dropna(subset=[demand_col], inplace=True)
    if len(processed_df) < initial_rows:
        st.warning(f"🗑️ Removed {initial_rows - len(processed_df)} rows with non-numeric or missing demand values.")
    else:
        st.success("✅ No non-numeric or missing demand values found to remove.")
    daily_total_demand = processed_df.groupby(day_col)[demand_col].sum().reset_index()
    daily_total_demand.rename(columns={demand_col: 'Total_Demand'}, inplace=True)
    st.info(f"📊 Aggregated total demand for {len(daily_total_demand)} unique days.")
    return daily_total_demand

# Chapter 3
def filter_and_sort_demand(daily_demand_df, max_demand_threshold):
    if daily_demand_df is None or daily_demand_df.empty:
        st.warning("⚠️ No data to filter or sort.")
        return None
    initial_rows = len(daily_demand_df)
    filtered_df = daily_demand_df[daily_demand_df['Total_Demand'] <= max_demand_threshold].copy()
    if len(filtered_df) < initial_rows:
        st.info(f"✂️ Successfully filtered out {initial_rows - len(filtered_df)} days with Total_Demand > {max_demand_threshold}.")
    else:
        st.success("✅ No demand values found above the threshold to remove.")
    sorted_df = filtered_df.sort_values(by='Total_Demand', ascending=True)
    return sorted_df

# Chapter 4
def calculate_demand_frequency_and_probability(daily_demand_df):
    if daily_demand_df is None or 'Total_Demand' not in daily_demand_df.columns or daily_demand_df.empty:
        st.warning("⚠️ Cannot calculate frequency and probability: DataFrame is empty.")
        return None
    demand_counts = daily_demand_df['Total_Demand'].value_counts().reset_index()
    demand_counts.columns = ['Total_Demand', 'Frequency']
    demand_counts['Probability'] = demand_counts['Frequency'] / demand_counts['Frequency'].sum()
    demand_counts = demand_counts.sort_values(by='Total_Demand').reset_index(drop=True)
    return demand_counts

def analyze_and_visualize_distribution(daily_demand_df, title_suffix=""):
    if daily_demand_df is None or 'Total_Demand' not in daily_demand_df.columns or daily_demand_df.empty:
        return
    st.subheader("📊 Descriptive Statistics")
    st.dataframe(daily_demand_df['Total_Demand'].describe())
    col1, col2 = st.columns(2)
    col1.metric("Skewness", f"{daily_demand_df['Total_Demand'].skew():.4f}")
    col2.metric("Kurtosis", f"{daily_demand_df['Total_Demand'].kurt():.4f}")
    st.subheader("📈 Demand Distribution Histogram")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(daily_demand_df['Total_Demand'], kde=True, bins=30, color='skyblue', edgecolor='black', ax=ax)
    ax.set_title(f'Distribution of Total Daily Demand {title_suffix}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Total Daily Demand (Units)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.grid(axis='y', alpha=0.75, linestyle='--')
    st.pyplot(fig)

# Chapter 5
@st.cache_data
def calculate_demand_during_lead_time_probability(demand_prob_table, lead_time_days=2):
    if demand_prob_table is None or demand_prob_table.empty:
        return None
    demands = demand_prob_table['Total_Demand'].values
    probs = demand_prob_table['Probability'].values
    max_daily_demand = int(demands.max())
    full_probs = np.zeros(max_daily_demand + 1)
    full_probs[demands.astype(int)] = probs
    convolved_probs = full_probs
    for _ in range(lead_time_days - 1):
        convolved_probs = np.convolve(convolved_probs, full_probs)
    final_demands = np.arange(len(convolved_probs))
    final_lead_time_dist = pd.DataFrame({'Demand_During_LeadTime': final_demands, 'Probability': convolved_probs})
    final_lead_time_dist = final_lead_time_dist[final_lead_time_dist['Probability'] > 1e-9].copy()
    final_lead_time_dist['CSL'] = final_lead_time_dist['Probability'].cumsum()
    return final_lead_time_dist.sort_values(by='Demand_During_LeadTime').reset_index(drop=True)

# Chapter 6
@st.cache_data
def calculate_expected_shortage(_ddlt_prob_table):
    if _ddlt_prob_table is None or _ddlt_prob_table.empty:
        return None
    df = _ddlt_prob_table.copy()
    df = df.sort_values('Demand_During_LeadTime').reset_index(drop=True)
    df['R'] = df['Demand_During_LeadTime']
    df['x_fx'] = df['Demand_During_LeadTime'] * df['Probability']
    sum_xfx_total = df['x_fx'].sum()
    df['Sum_xfx_Rplus1'] = sum_xfx_total - df['x_fx'].cumsum()
    df['Sum_R_fx_Rplus1'] = df['R'] * (1 - df['CSL'])
    df['E_S'] = (df['Sum_xfx_Rplus1'] - df['Sum_R_fx_Rplus1']).clip(lower=0)
    return df[['R', 'Probability', 'CSL', 'E_S']]

# >>> Chapter 7: Function for (Q, R) Iterative Optimization <<<
def find_optimal_qr(ddlt_with_shortage_table, daily_avg_demand, avg_lead_time,
                    cp_ordering_cost, product_cost, h_percent_annual, s_percent,
                    service_level_type='backlog', convergence_tolerance=0.001, max_iterations=100):
    """
    Finds the optimal Order Quantity (Q) and Reorder Point (R) using an iterative approach.
    """
    if ddlt_with_shortage_table is None or ddlt_with_shortage_table.empty:
        st.error("⚠️ Invalid DDLT table provided for Q,R optimization.")
        return None

    # --- Initialization ---
    days_per_year = 365
    Cp = cp_ordering_cost
    D_annual = daily_avg_demand * days_per_year
    mu_DL = daily_avg_demand * avg_lead_time
    Ch_annual = product_cost * h_percent_annual
    Cs_per_unit = product_cost * s_percent

    # --- Step 1: Compute Initial Q (EOQ) ---
    Q = np.sqrt((2 * D_annual * Cp) / Ch_annual)

    # --- Prepare for iteration ---
    Q_old = 0.0
    iteration = 0
    optimization_history = []
    ddlt_table_sorted = ddlt_with_shortage_table.sort_values(by='R').reset_index(drop=True)

    while abs(Q - Q_old) > convergence_tolerance and iteration < max_iterations:
        Q_old = Q
        iteration += 1

        # --- Step 2: Determine Optimal CSL ---
        if service_level_type == 'backlog':
            csl_optimal = 1 - (Q * Ch_annual) / (D_annual * Cs_per_unit)
        elif service_level_type == 'lost_sales':
            csl_optimal = (D_annual * Cs_per_unit) / ((D_annual * Cs_per_unit) + (Q * Ch_annual))
        else:
            st.error(f"❌ Error: Invalid service_level_type received: '{service_level_type}'. Check code logic.")
            return None
        csl_optimal = max(0, min(1, csl_optimal))

        # --- Step 3: Find Reorder Point (R) ---
        r_candidates = ddlt_table_sorted[ddlt_table_sorted['CSL'] >= csl_optimal]
        if not r_candidates.empty:
            R = r_candidates.iloc[0]['R']
        else:
            R = ddlt_table_sorted['R'].max()

        # --- Step 4: Compute Costs ---
        es_at_R_row = ddlt_table_sorted.loc[ddlt_table_sorted['R'] == R, 'E_S']
        es_at_R = es_at_R_row.iloc[0] if not es_at_R_row.empty else 0.0

        ordering_cost = (D_annual / Q) * Cp
        if service_level_type == 'backlog':
            holding_cost = ((Q / 2) + R - mu_DL) * Ch_annual
        else: # Lost Sales
            holding_cost = ((Q / 2) + R - mu_DL + es_at_R) * Ch_annual
        shortage_cost = (D_annual / Q) * Cs_per_unit * es_at_R
        TAC = ordering_cost + holding_cost + shortage_cost

        # --- LOG HISTORY ---
        optimization_history.append({
            'Iteration': iteration,
            'Q_old': Q_old,
            'CSL_optimal': csl_optimal,
            'R_found': R,
            'E_S_at_R': es_at_R,
            'TAC': TAC
        })

        # --- Step 5: Recompute Q ---
        Q = np.sqrt((2 * D_annual * (Cp + Cs_per_unit * es_at_R)) / Ch_annual)

    final_q = Q
    final_r = R
    final_tac = TAC

    if iteration >= max_iterations:
        st.warning(f"⚠️ Max iterations ({max_iterations}) reached. Solution might not have fully converged.")
    else:
        st.success(f"🎉 Convergence achieved in {iteration} iterations!")

    return {
        'optimal_Q': final_q,
        'optimal_R': final_r,
        'min_TAC': final_tac,
        'convergence_iterations': iteration,
        'history': pd.DataFrame(optimization_history)
    }

# --- Streamlit App UI ---
st.title("🎓 Master's Independent Study: Demand & Inventory Analysis")
st.markdown("#### *Advised by: DR. JIRACHAI BUDDHAKULSOMSIRI*")
st.markdown("---")
st.markdown("Welcome! This app guides you through analyzing historical demand data to determine optimal inventory policies.")

st.sidebar.image("https://admissions.siit.tu.ac.th/wp-content/uploads/2023/06/cropped-TU-SIIT1992-01.png", width=250)
st.sidebar.header("⚙️ Control Panel")
uploaded_file = st.sidebar.file_uploader("1. Upload Raw Data (CSV or XLSX)", type=['csv', 'xlsx'])

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}

if uploaded_file is not None:
    if 'file_name' not in st.session_state.processed_data or st.session_state.processed_data['file_name'] != uploaded_file.name:
        with st.spinner("Loading and inspecting data..."):
            st.session_state.processed_data['raw_data_df'] = load_and_inspect_data(uploaded_file)
            st.session_state.processed_data['file_name'] = uploaded_file.name

    if st.session_state.processed_data.get('raw_data_df') is not None:
        st.sidebar.markdown("---")
        st.sidebar.header("2. Analysis Parameters")

        max_demand_threshold = st.sidebar.slider("Demand Outlier Threshold", 300, 1500, 700, 10, help="Any daily demand > this value will be excluded.")
        lead_time_days = st.sidebar.slider("Lead Time (Days)", 1, 10, 2, 1, help="Fixed lead time for order replenishment.")

        raw_df = st.session_state.processed_data['raw_data_df']
        agg_df = preprocess_and_aggregate_demand(raw_df, 'Date', 'Units Sold')
        final_demand_df = filter_and_sort_demand(agg_df, max_demand_threshold)
        st.session_state.processed_data['final_demand_df'] = final_demand_df

        st.header("Chapter 1-4: Demand Analysis")
        with st.expander("Show/Hide Demand Analysis Details", expanded=True):
            if final_demand_df is not None:
                analyze_and_visualize_distribution(final_demand_df, title_suffix=f"(Max Demand ≤ {max_demand_threshold})")

        st.header("Chapter 5-6: Lead Time Demand & Expected Shortage")
        with st.expander("Show/Hide Lead Time & Shortage Tables", expanded=True):
            with st.spinner(f"Calculating distribution for a {lead_time_days}-day lead time..."):
                demand_prob_table = calculate_demand_frequency_and_probability(final_demand_df)
                ddlt_prob_table = calculate_demand_during_lead_time_probability(demand_prob_table, lead_time_days)

            if ddlt_prob_table is not None:
                st.session_state.processed_data['ddlt_prob_table'] = ddlt_prob_table
                st.subheader(f"Demand During Lead Time ({lead_time_days} days)")
                st.dataframe(ddlt_prob_table)

                with st.spinner("Calculating Expected Shortage (E(S))..."):
                    final_ddlt_with_shortage = calculate_expected_shortage(ddlt_prob_table)
                    st.session_state.processed_data['final_ddlt_with_shortage'] = final_ddlt_with_shortage

                st.subheader("Expected Shortage (E(S)) vs. Reorder Point (R)")
                st.dataframe(final_ddlt_with_shortage)
            else:
                 st.error("Could not calculate DDLT or E(S) tables.")

        st.sidebar.markdown("---")
        st.sidebar.header("3. Cost & Policy Parameters")
        with st.sidebar.form(key='cost_form'):
            st.subheader("Enter Cost Values")
            cp_cost = st.number_input("Ordering Cost (Cp) / order", 0.0, value=10.0, step=1.0)
            product_cost = st.number_input("Product Cost / unit", 0.0, value=50.0, step=1.0)
            h_percent = st.slider("Annual Holding Rate (h) %", 0, 100, 10, help="As an annual % of product cost.")
            s_percent = st.slider("Shortage Cost Rate (s) %", 0, 100, 30, help="As a % of product cost per unit.")

            st.subheader("Select Inventory Case")
            case_type = st.radio("Shortage Scenario", ('Lost Sales', 'Backlog'), help="Determines the cost formula used for shortages.")
            
            submitted = st.form_submit_button("🚀 Run (Q, R) Optimization")

        if submitted:
            if 'final_ddlt_with_shortage' in st.session_state.processed_data:
                st.header("Chapter 7: Optimal (Q, R) Policy")

                final_ddlt = st.session_state.processed_data['final_ddlt_with_shortage']
                daily_avg_demand = st.session_state.processed_data['final_demand_df']['Total_Demand'].mean()
                
                service_level_type = case_type.lower().replace(' ', '_')

                with st.spinner(f"Finding Optimal Q and R for '{case_type}' case..."):
                    qr_results = find_optimal_qr(
                        ddlt_with_shortage_table=final_ddlt,
                        daily_avg_demand=daily_avg_demand,
                        avg_lead_time=lead_time_days,
                        cp_ordering_cost=cp_cost,
                        product_cost=product_cost,
                        h_percent_annual=(h_percent / 100.0),
                        s_percent=(s_percent / 100.0),
                        service_level_type=service_level_type,
                        convergence_tolerance=0.01,
                        max_iterations=50
                    )

                if qr_results:
                    st.balloons()
                    st.subheader(f"Optimal (Q, R) System Parameters ({case_type} Case)")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Optimal Order Quantity (Q)", f"{qr_results['optimal_Q']:,.0f} units")
                    col2.metric("Optimal Reorder Point (R)", f"{qr_results['optimal_R']:,.0f} units")
                    col3.metric("Minimum Annual Cost (TAC)", f"{qr_results['min_TAC']:,.2f} THB")
                    
                    with st.expander("🔎 View Optimization Convergence Details", expanded=True):
                        history_df = qr_results['history']
                        
                        st.subheader("Convergence Plot")
                        fig, ax1 = plt.subplots(figsize=(12, 6))

                        color = 'tab:blue'
                        ax1.set_xlabel('Iteration', fontsize=14)
                        ax1.set_ylabel('Order Quantity (Q)', color=color, fontsize=14)
                        ax1.plot(history_df['Iteration'], history_df['Q_old'], color=color, marker='o', linestyle='-', label='Order Quantity (Q)')
                        ax1.tick_params(axis='y', labelcolor=color)
                        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

                        ax2 = ax1.twinx()
                        color = 'tab:red'
                        ax2.set_ylabel('Total Annual Cost (TAC)', color=color, fontsize=14)
                        ax2.plot(history_df['Iteration'], history_df['TAC'], color=color, marker='x', linestyle='--', label='Total Annual Cost (TAC)')
                        ax2.tick_params(axis='y', labelcolor=color)

                        fig.suptitle('Q and TAC Convergence Over Iterations', fontsize=16, fontweight='bold')
                        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                        st.pyplot(fig)

                        st.subheader("Iteration History Table")
                        st.dataframe(history_df.style.format({
                            'Q_old': '{:,.2f}',
                            'CSL_optimal': '{:.4f}',
                            'R_found': '{:,.0f}',
                            'E_S_at_R': '{:.4f}',
                            'TAC': '{:,.2f}'
                        }))
            else:
                st.error("❌ Cannot run optimization. Data processing is not complete. Please ensure data is loaded.")
else:
    st.info("👋 Welcome! Please upload your demand data (CSV or XLSX) using the sidebar to begin the analysis.")