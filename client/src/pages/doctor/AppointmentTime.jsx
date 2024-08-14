import React, { useState } from "react";

const AppointmentTime = (appointmentTime) => {
  const [selectedTime, setSelectedTime] = useState();

  const handleSelect = (e) => {
    setSelectedTime(e.target.value);
    console.log(e.target.value);
  };
  return (
    <>
      <div>
        <label htmlFor="appointment-time">Select appointment time:</label>
        <select ref={appointmentTime}
          style={{
            marginLeft: "10px",
            
          }}
          id="appointment-time"
          value={selectedTime}
          onChange={(e) => handleSelect(e)}
        >
          <option value="">Select</option>
          <option value="6:00am">6:00 -7:00</option>
          <option value="7:00am">7:00 -8:00</option>
          <option value="8:00am">8:00 -9:00</option>

          <option value="9:00am">9:00 -10:00</option>
          <option value="10:00am">10:00 - 11:00</option>
          <option value="11:00am">11:00 - 12:00</option>
          <option value="12:00pm">12:00 -1:00</option>
          <option value="1:00pm">1:00 - 2:00</option>
          <option value="2:00pm">2:00 - 3:00</option>
          <option value="3:00pm">3:00 - 4:00</option>
          <option value="4:00pm">4:00 - 5:00</option>
          <option value="5:00pm">5:00 - 6:00</option>
        </select>
      </div>
    </>
  );
};

export default AppointmentTime;
