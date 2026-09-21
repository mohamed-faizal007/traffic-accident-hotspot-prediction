import pandas as pd
import folium

from pathlib import Path
from folium.plugins import MarkerCluster


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTIONS_PATH = Path(
    "outputs/final_hotspot_predictions.csv"
)

OUTPUT_DIR = Path("outputs")

MAP_PATH = (
    OUTPUT_DIR
    / "accident_hotspot_risk_map.html"
)


# ============================================================
# MAP CONFIGURATION
# ============================================================

MAX_MEDIUM_RISK_GRIDS = 1000

MAX_HIGH_RISK_GRIDS = 3000

MAX_CRITICAL_RISK_GRIDS = 3000


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_section(title):

    print("\n" + "=" * 60)

    print(title)

    print("=" * 60)


# ============================================================
# STEP 1: LOAD PREDICTIONS
# ============================================================

def load_predictions():

    print_section(
        "STEP 1: LOADING FINAL HOTSPOT PREDICTIONS"
    )

    df = pd.read_csv(
        PREDICTIONS_PATH,
        low_memory=False
    )

    print(
        f"Total prediction records: {len(df):,}"
    )

    print(
        f"Total columns: {len(df.columns)}"
    )

    print(
        f"Dataset loaded from:\n{PREDICTIONS_PATH}"
    )

    return df


# ============================================================
# STEP 2: AGGREGATE MONTHLY PREDICTIONS BY GRID
# ============================================================

def aggregate_grid_predictions(df):

    print_section(
        "STEP 2: AGGREGATING PREDICTIONS BY SPATIAL GRID"
    )

    print(
        "Multiple monthly predictions will be combined"
    )

    print(
        "into one risk record per spatial grid."
    )

    grid_df = (

        df.groupby("grid_id")

        .agg(

            grid_latitude=(
                "grid_latitude",
                "mean"
            ),

            grid_longitude=(
                "grid_longitude",
                "mean"
            ),

            max_prediction_probability=(
                "prediction_probability",
                "max"
            ),

            average_prediction_probability=(
                "prediction_probability",
                "mean"
            ),

            predicted_hotspot_months=(
                "predicted_hotspot",
                "sum"
            ),

            total_months=(
                "year_month",
                "count"
            )

        )

        .reset_index()

    )


    print(
        f"\nUnique spatial grids: {len(grid_df):,}"
    )

    print(
        "Monthly predictions successfully aggregated."
    )

    return grid_df


# ============================================================
# STEP 3: CREATE GRID RISK LEVELS
# ============================================================

def create_grid_risk_levels(grid_df):

    print_section(
        "STEP 3: CREATING GRID RISK LEVELS"
    )

    def assign_risk(probability):

        if probability >= 0.90:

            return "Critical"

        elif probability >= 0.70:

            return "High"

        elif probability >= 0.50:

            return "Medium"

        else:

            return "Low"


    grid_df[
        "risk_level"
    ] = grid_df[
        "max_prediction_probability"
    ].apply(
        assign_risk
    )


    print(
        "Risk levels created using maximum"
    )

    print(
        "predicted probability across months."
    )


    print(
        "\nRisk Level Distribution:"
    )

    print(
        grid_df[
            "risk_level"
        ]
        .value_counts()
    )


    return grid_df


# ============================================================
# STEP 4: SELECT MAP DATA
# ============================================================

def select_map_data(grid_df):

    print_section(
        "STEP 4: SELECTING DATA FOR MAP VISUALIZATION"
    )


    critical_df = (

        grid_df[
            grid_df["risk_level"] == "Critical"
        ]

        .sort_values(
            "max_prediction_probability",
            ascending=False
        )

        .head(
            MAX_CRITICAL_RISK_GRIDS
        )

    )


    high_df = (

        grid_df[
            grid_df["risk_level"] == "High"
        ]

        .sort_values(
            "max_prediction_probability",
            ascending=False
        )

        .head(
            MAX_HIGH_RISK_GRIDS
        )

    )


    medium_df = (

        grid_df[
            grid_df["risk_level"] == "Medium"
        ]

        .sort_values(
            "max_prediction_probability",
            ascending=False
        )

        .head(
            MAX_MEDIUM_RISK_GRIDS
        )

    )


    print(
        f"Critical risk grids selected: "
        f"{len(critical_df):,}"
    )

    print(
        f"High risk grids selected: "
        f"{len(high_df):,}"
    )

    print(
        f"Medium risk grids selected: "
        f"{len(medium_df):,}"
    )


    return (

        critical_df,

        high_df,

        medium_df

    )


# ============================================================
# STEP 5: CREATE BASE MAP
# ============================================================

def create_base_map(grid_df):

    print_section(
        "STEP 5: CREATING INTERACTIVE MAP"
    )


    center_latitude = (

        grid_df[
            "grid_latitude"
        ].mean()

    )


    center_longitude = (

        grid_df[
            "grid_longitude"
        ].mean()

    )


    print(
        f"Map center latitude: "
        f"{center_latitude:.6f}"
    )

    print(
        f"Map center longitude: "
        f"{center_longitude:.6f}"
    )


    accident_map = folium.Map(

        location=[

            center_latitude,

            center_longitude

        ],

        zoom_start=6,

        tiles="CartoDB positron"

    )


    print(
        "\nInteractive base map created."
    )


    return accident_map


# ============================================================
# STEP 6: ADD CRITICAL RISK LOCATIONS
# ============================================================

def add_critical_risk_layer(

    accident_map,

    critical_df

):

    print_section(
        "STEP 6: ADDING CRITICAL RISK LOCATIONS"
    )


    critical_layer = folium.FeatureGroup(

        name="Critical Risk"

    )


    cluster = MarkerCluster().add_to(

        critical_layer

    )


    for _, row in critical_df.iterrows():


        popup_text = f"""

        <b>ACCIDENT RISK: CRITICAL</b>

        <br><br>

        <b>Grid ID:</b>
        {row['grid_id']}

        <br>

        <b>Maximum Risk Probability:</b>
        {row['max_prediction_probability']:.4f}

        <br>

        <b>Average Risk Probability:</b>
        {row['average_prediction_probability']:.4f}

        <br>

        <b>Predicted Hotspot Months:</b>
        {int(row['predicted_hotspot_months'])}

        <br>

        <b>Total Months Analyzed:</b>
        {int(row['total_months'])}

        """


        folium.CircleMarker(

            location=[

                row[
                    "grid_latitude"
                ],

                row[
                    "grid_longitude"
                ]

            ],

            radius=6,

            popup=folium.Popup(

                popup_text,

                max_width=300

            ),

            tooltip="Critical Accident Risk",

            color="red",

            fill=True,

            fill_color="red",

            fill_opacity=0.8

        ).add_to(cluster)


    critical_layer.add_to(

        accident_map

    )


    print(
        f"Critical locations added: "
        f"{len(critical_df):,}"
    )


# ============================================================
# STEP 7: ADD HIGH RISK LOCATIONS
# ============================================================

def add_high_risk_layer(

    accident_map,

    high_df

):

    print_section(
        "STEP 7: ADDING HIGH RISK LOCATIONS"
    )


    high_layer = folium.FeatureGroup(

        name="High Risk"

    )


    cluster = MarkerCluster().add_to(

        high_layer

    )


    for _, row in high_df.iterrows():


        popup_text = f"""

        <b>ACCIDENT RISK: HIGH</b>

        <br><br>

        <b>Grid ID:</b>
        {row['grid_id']}

        <br>

        <b>Maximum Risk Probability:</b>
        {row['max_prediction_probability']:.4f}

        <br>

        <b>Average Risk Probability:</b>
        {row['average_prediction_probability']:.4f}

        <br>

        <b>Predicted Hotspot Months:</b>
        {int(row['predicted_hotspot_months'])}

        """


        folium.CircleMarker(

            location=[

                row[
                    "grid_latitude"
                ],

                row[
                    "grid_longitude"
                ]

            ],

            radius=5,

            popup=folium.Popup(

                popup_text,

                max_width=300

            ),

            tooltip="High Accident Risk",

            color="orange",

            fill=True,

            fill_color="orange",

            fill_opacity=0.7

        ).add_to(cluster)


    high_layer.add_to(

        accident_map

    )


    print(
        f"High risk locations added: "
        f"{len(high_df):,}"
    )


# ============================================================
# STEP 8: ADD MEDIUM RISK LOCATIONS
# ============================================================

def add_medium_risk_layer(

    accident_map,

    medium_df

):

    print_section(
        "STEP 8: ADDING MEDIUM RISK LOCATIONS"
    )


    medium_layer = folium.FeatureGroup(

        name="Medium Risk"

    )


    cluster = MarkerCluster().add_to(

        medium_layer

    )


    for _, row in medium_df.iterrows():


        popup_text = f"""

        <b>ACCIDENT RISK: MEDIUM</b>

        <br><br>

        <b>Grid ID:</b>
        {row['grid_id']}

        <br>

        <b>Maximum Risk Probability:</b>
        {row['max_prediction_probability']:.4f}

        <br>

        <b>Average Risk Probability:</b>
        {row['average_prediction_probability']:.4f}

        """


        folium.CircleMarker(

            location=[

                row[
                    "grid_latitude"
                ],

                row[
                    "grid_longitude"
                ]

            ],

            radius=4,

            popup=folium.Popup(

                popup_text,

                max_width=300

            ),

            tooltip="Medium Accident Risk",

            color="yellow",

            fill=True,

            fill_color="yellow",

            fill_opacity=0.6

        ).add_to(cluster)


    medium_layer.add_to(

        accident_map

    )


    print(
        f"Medium risk locations added: "
        f"{len(medium_df):,}"
    )


# ============================================================
# STEP 9: ADD MAP LEGEND
# ============================================================

def add_map_legend(accident_map):

    legend_html = """

    <div style="

        position: fixed;

        bottom: 50px;

        left: 50px;

        width: 220px;

        height: 180px;

        background-color: white;

        border: 2px solid grey;

        z-index: 9999;

        font-size: 14px;

        padding: 15px;

    ">

    <b>ACCIDENT HOTSPOT RISK</b>

    <br><br>

    <span style="color:red;">
    ●
    </span>

    Critical Risk

    <br>

    <span style="color:orange;">
    ●
    </span>

    High Risk

    <br>

    <span style="color:#c9b800;">
    ●
    </span>

    Medium Risk

    <br><br>

    <b>Model:</b>

    Advanced Random Forest

    <br>

    <b>Threshold:</b>

    0.90

    </div>

    """


    accident_map.get_root().html.add_child(

        folium.Element(
            legend_html
        )

    )


# ============================================================
# STEP 10: ADD LAYER CONTROL
# ============================================================

def add_layer_control(accident_map):

    folium.LayerControl(

        collapsed=False

    ).add_to(

        accident_map

    )


# ============================================================
# STEP 11: SAVE MAP
# ============================================================

def save_map(accident_map):

    print_section(
        "STEP 9: SAVING INTERACTIVE HOTSPOT MAP"
    )


    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    accident_map.save(

        MAP_PATH

    )


    print(
        "Interactive hotspot map saved successfully."
    )

    print(
        f"\nMap path:\n{MAP_PATH}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n")

    print("#" * 60)

    print(
        "TRAFFIC ACCIDENT HOTSPOT PREDICTION"
    )

    print(
        "INTERACTIVE SPATIAL RISK MAP"
    )

    print("#" * 60)


    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    df = load_predictions()


    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    grid_df = (

        aggregate_grid_predictions(df)

    )


    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    grid_df = (

        create_grid_risk_levels(grid_df)

    )


    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    (

        critical_df,

        high_df,

        medium_df

    ) = select_map_data(

        grid_df

    )


    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    accident_map = (

        create_base_map(grid_df)

    )


    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    add_critical_risk_layer(

        accident_map,

        critical_df

    )


    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------

    add_high_risk_layer(

        accident_map,

        high_df

    )


    # --------------------------------------------------------
    # STEP 8
    # --------------------------------------------------------

    add_medium_risk_layer(

        accident_map,

        medium_df

    )


    # --------------------------------------------------------
    # STEP 9
    # --------------------------------------------------------

    add_map_legend(

        accident_map

    )


    # --------------------------------------------------------
    # STEP 10
    # --------------------------------------------------------

    add_layer_control(

        accident_map

    )


    # --------------------------------------------------------
    # STEP 11
    # --------------------------------------------------------

    save_map(

        accident_map

    )


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_section(
        "HOTSPOT MAP CREATION COMPLETE"
    )


    print(
        "The interactive accident risk map"
    )

    print(
        "contains:"
    )

    print(
        "- Critical risk locations"
    )

    print(
        "- High risk locations"
    )

    print(
        "- Medium risk locations"
    )

    print(
        "- Prediction probabilities"
    )

    print(
        "- Predicted hotspot frequency"
    )

    print(
        "- Interactive location popups"
    )

    print(
        "- Layer controls"
    )


    print(
        "\nOpen the following file"
    )

    print(
        "in your browser:"
    )

    print(
        MAP_PATH
    )


if __name__ == "__main__":

    main()