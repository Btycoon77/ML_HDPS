const joi = require('joi');

const doctorSchema = joi.object({
    firstName:joi.string().required(),
    lastName:joi.string().required(),
    phone:joi.number().required(),
    email: joi.string().email({ minDomainSegments: 2, tlds: { allow: ['com', 'net'] } }).required(),
    website: joi.string().required(),
    address:joi.string().required(),
    specialization: joi.string().required(),
    experience: joi.string().required(),
    feesPerConsultation: joi.number().required(),

});


const user = {
    name: 'John Doe',
    email: 'john.doe@example.com',
   
  };
  
  const result = doctorSchema.validate(user);
  
  if (result.error) {
    console.log(result.error.details[0].message);
  } else {
    console.log('User object is valid!');
  }