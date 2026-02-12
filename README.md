<h2 align="center">
  Explainability in a chaotic system – Application to weather forecasting
</h2>

<p align="center">
  <img src="era5_visuals/gifs/alex_europe.gif" width="400">
</p>


## 💡 Overview

This repository contains all the code developed as part of a CentraleSupélec project conducted in partnership with HeadMind Partners, focusing on the explainability of weather forecasting models. Specifically, we study precipitation prediction over Europe within a 6-hour forecasting horizon. Although some experiments were carried out with longer time horizons, their predictive performance was significantly lower; as a result, we chose not to include their explainability analyses in this repository. All models were trained using the ERA5 dataset from WeatherBench2.

The project is structured into two main phases:

1. **Precipitation prediction.**
We provide scripts to download and preprocess the data, train two types of models (U-Net and ConvLSTM), and evaluate their performance. More detailed information about the files and workflows is provided in a later section.

2. **Prediction explainability.**
We implement permutation-based methods and integrated gradients, combined with various aggregation strategies, to extract insights into the most influential input variables and time steps. These methods allow us to analyze which pixels contribute most to individual predictions, identify globally important features, explore patterns that are consistent with meteorological knowledge, and more. More detailed information about the explainability pipeline and related files is provided in a later section.


## 📦 Getting Started

To get a local copy of this project up and running, follow these steps.

### 🚀 Prerequisites

- **Node.js** (v16.x or higher) and **npm** or **yarn**.
- **Npm** If you prefer using npm for package management and running scripts.
- **PostgreSQL** (or another supported SQL database).

## 🛠️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/sumonta056/readme-template.git
   cd readme-template
   ```

2. **Install dependencies:**

   Using Npm:

   ```bash
   npm install
   ```

3. **Set up environment variables:**

   Create a `.env` file in the root directory and add the following variables:

   ```env
   NEXT_PUBLIC_APP_URL=http://localhost:3000

   #database
   DATABASE_URL=your_database_url
   DATABASE_SECRET=your_database_secret
   DRIZZLE_DATABASE_URL=your_database_url_for_drizzle

   #auth
   AUTH_SECRET=any_random_secret
   ```

4. **Run database migrations:**

   Ensure your database is running and then run:

   ```bash
   npm run drizzle-kit migrate
   ```

5. **Start the development server:**

   ```bash
   npm run dev
   ```

## 📖 Usage

### Repository structure description

### Visualizing some variables

### Downloading the dataset

### Training a weather forecasting model

### Explaining a pretrained model




## 🤝 Authors

This repository was created and equally contributed to by :
- Louisa Arfib : [https://github.com/arfiblouisa](https://github.com/arfiblouisa)
- Manon Arfib : [https://github.com/manonarfib](https://github.com/manonarfib)
- Nathan Morin : [https://github.com/Nathan9842](https://github.com/Nathan9842)

## Acknowledgment
A huge thank you to Florestan Fontaine for his help and valuable advice.