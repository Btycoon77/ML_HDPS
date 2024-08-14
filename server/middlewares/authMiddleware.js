const jwt = require("jsonwebtoken");

const authMiddlware = async (req, res, next) => {
  try {
    const token = req.headers["authorization"].split(" ")[1];

    // there is naming convention in token such as Bearer fhdkflja;sdlfk(token) -- if we split it we get in the form of array since we need to access the token value only;
    
    jwt.verify(token, process.env.JWT_SECRET, (err, decode) => {
      if (err) {
        return res.status(200).json({ message: "Auth failed", success: false });
      } else {
        req.body.userId = decode.id;
        next();
      }
    });
  } catch (error) {
    console.log(error);
    res.status(401).send({ messsage: "Auth failed", success: false });
  }
};

module.exports = authMiddlware;
