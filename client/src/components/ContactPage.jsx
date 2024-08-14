import React from "react";

const ContactPage = () => {
  const containerStyle = {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    height: "100vh",
    backgroundColor: "#f2f2f2",
  };

  const contentStyle = {
    display: "flex",
    flexDirection: "row",
    justifyContent: "space-between",
    width: "80%",
    padding: "20px",
    backgroundColor: "#fff",
    borderRadius: "10px",
    boxShadow: "0 2px 10px rgba(0, 0, 0, 0.1)",
  };

  const leftSideStyle = {
    display: "flex",
    flexDirection: "column",
    gap: "20px",
  };

  const rightSideStyle = {
    display: "flex",
    flexDirection: "column",
    gap: "20px",
  };

  const topicTextStyle = {
    fontSize: "24px",
    fontWeight: "bold",
  };

  const inputStyle = {
    padding: "10px",
    border: "1px solid #ccc",
    borderRadius: "5px",
    width: "100%",
  };

  const submitButtonStyle = {
    padding: "10px 20px",
    backgroundColor: "blue",
    color: "#fff",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
  };

  return (
    <div style={containerStyle}>
      <div style={contentStyle}>
        <div style={leftSideStyle}>
          <div className="address details">
            <i className="fas fa-map-marker-alt"></i>
            <div className="topic">Address</div>
            <div className="text-one">Kirtipur</div>
            <div className="text-two">Kathmandu</div>
          </div>
          <div className="phone details">
            <i className="fas fa-phone-alt"></i>
            <div className="topic">Phone</div>
            <div className="text-one">+977 9869068956</div>
            <div className="text-two">+977 9840135234</div>
          </div>
          <div className="email details">
            <i className="fas fa-envelope"></i>
            <div className="topic">Email</div>
            <div className="text-one">btycoon77@gmail.com</div>
            <div className="text-two">ehealthcare@gmail.com</div>
          </div>
        </div>
        <div style={rightSideStyle}>
          <div style={topicTextStyle}>Send us a message</div>
          <p>
            If you have any work from me or any types of queries related to
            your health, please feel free to send us a message from here. It's
            our pleasure to help you.
          </p>
          <form action="#" method="post">
            <div className="input-box">
              <input
                style={inputStyle}
                type="text"
                placeholder="Enter your name"
                name="name"
              />
            </div>

            <div className="input-box">
              <input
                style={inputStyle}
                type="text"
                placeholder="Enter your email"
                name="email"
              />
            </div>
            <div className="input-box">
              <textarea
                style={inputStyle}
                name="message"
                id="message"
                cols="30"
                rows="5"
                placeholder="Your message"
              ></textarea>
            </div>
            <div className="button">
              <input
                style={submitButtonStyle}
                type="submit"
                name="submit"
                value="Send Now"
              />
            </div>
          </form>
        </div>
      </div>
      <a href="/index">Back to Homepage</a>
    </div>
  );
};

export default ContactPage;
