const Joi = require('joi');

const userSchema = Joi.object({
    name: Joi.string().required(),
    email: Joi.string().email({ minDomainSegments: 2, tlds: { allow: ['com', 'net'] } }).required(),
    password: Joi.string().pattern(new RegExp('^[a-zA-Z0-9]{3,30}$')).required(),

});

// const user = {
//     name: 'John Doe',
//     email: 'john.doe@example.com',
//     password: 'mypassword123'
//   };
  
//   const result = userSchema.validate(user);
  
//   if (result.error) {
//     console.log(result.error.details[0].message);
//   } else {
//     console.log('User object is valid!');
//   }

module.exports = userSchema;