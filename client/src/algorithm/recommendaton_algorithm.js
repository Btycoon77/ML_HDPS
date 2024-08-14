// Define the features of doctors

const doctor_features = {
  specialization: [
    "Pediatrician",
    "Cardiologist",
    "Dermatologist",
    "Gynecologist",
    "Orthopedist",
  ],
  location: ["New York", "Los Angeles", "Chicago", "Houston", "Miami"],
  availability: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
//   ratings: [3.5, 4.0, 4.5, 5.0], 
};

// Collect user information
const user_info = {
  specialization: "Pediatrician",
  location: "New York",
  availability: "2:00",
//   ratings: 4.5,
};

// Define a list of available doctors
const available_doctors = [
  {
    name: "Dr. Ram",
    specialization: "Pediatrician",
    location: "New York",
    availability: ["Monday", "Tuesday", "Thursday"],
    // ratings: 4.5,
  },
  {
    name: "Dr. Laxman",
    specialization: "Dermatologist",
    location: "New York",
    availability: ["Wednesday", "Thursday", "Friday"],
     // ratings: 4.0,
  },
  {
    name: "Dr. Sagar",
    specialization: "Gynecologist",
    location: "Chicago",
    availability: ["Tuesday", "Wednesday", "Thursday"],
    // ratings: 4.5,
  },
  {
    name: "Dr. Bibek",
    specialization: "Cardiologist",
    location: "Los Angeles",
    availability: ["Monday", "Wednesday", "Friday"],
    // ratings: 4.0,
  },
  {
    name: "Dr. Kimkardashain",
    specialization: "Orthopedist",
    location: "Houston",
    availability: ["Monday", "Tuesday", "Wednesday"],
    // ratings: 3.5,
  },
];

// Define a function to filter and rank doctors based on user preferences
function recommend_doctors(user_info, available_doctors) {
  // Filter doctors based on specialization, location, and availability
  const filtered_doctors = available_doctors.filter(
    (doctor) =>
      // doctor["specialization"] == user_info["specialization"] &&
      doctor["location"] == user_info["location"] &&
      doctor["availability"].includes(user_info["availability"])
  );

  // Calculate similarity score for each filtered doctor
  filtered_doctors.forEach((doctor) => {
    let similarity_score = 0;
    for (const feature in user_info) {
      if (doctor[feature] === user_info[feature]) {
        similarity_score += 1;
      }
    }
    doctor["similarity_score"] = similarity_score;
  });

  // Sort filtered doctors by similarity score and ratings
  const recommended_doctors = filtered_doctors.sort((a, b) => {
    if (a["similarity_score"] !== b["similarity_score"]) {
        // all the doctors will be sorted in descending order based on their similarity scores.
      return b["similarity_score"] - a["similarity_score"];
    } else {
        // all the doctors will be sorted in descending order based on their ratings;
    //   return b["ratings"] - a["ratings"];
    }
  });

  return recommended_doctors;
}

// Call the recommend_doctors function with user information and available doctors
const recommended_doctors = recommend_doctors(user_info, available_doctors);

// Print the recommended doctors
recommended_doctors.forEach((doctor) => console.log("here are some of the recommended doctors for you",doctor["name"]));
