import { createSlice } from "@reduxjs/toolkit";

export const userSlice = createSlice({
  name: "user",
  initialState: {
    user: null,
  },
  reducers: {
    setUser: (state, action) => {
      //  Whatever data gets filled in the state,it shows in the action.payload

      state.user = action.payload;
    },
  },
});

export const { setUser } = userSlice.actions;
