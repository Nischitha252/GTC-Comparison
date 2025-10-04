// import * as dataMap from '../fixtures/inputData.json'
// // const FooterCypress = () => {
//   describe('footer', () => {
//     beforeEach(() => {
//       cy.visit('http://localhost:3000');
//     });

//     it('footer links', () => {
//       cy.get('.footer-links a').should('have.length', dataMap.links.length);
//       dataMap.links.forEach(link => {
//         cy.get('.footer-links a').eq(link.index)
//           .should('have.attr', 'href')
//           .click();
//         cy.url().should('include');
//         cy.go('back'); 
//       });
//     });
//   });
// //   FooterCypress();

// // export default FooterCypress;
// // const FooterCypress = () => {

describe('Footer', () => {
  beforeEach(() => {
    cy.visit('http://localhost:3000'); 
  });

  it('should navigate to Contact Us page', () => {
    cy.get('.footer-links').contains('Contact Us').click();
    cy.url().should('include', '/contact');
  });

  it('should navigate to Feedback page', () => {
    cy.get('.footer-links').contains('Feedback').click();
    cy.url().should('include', '/feedback');
  });
});
