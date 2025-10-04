// describe('WelcomePage', () => {
//     beforeEach(() => {
//         cy.visit('http://localhost:3000');
//     });
//     it('Gets GTC comparision', () => {
//         cy.get('.cardMenu').click();
//         cy.get('.card').eq(0).click();
//         cy.get('.multipleBackButton').click();
//         cy.get('.card').eq(1).click();

//     });
// })

describe('WelcomePage', () => {
    beforeEach(() => {
        cy.visit('http://localhost:3000');
    });
    it('Gets GTC comparision', () => {
        cy.get('.cardMenu').click();
        // cy.get('.card').eq(0).click();
        cy.get('.multipleBackButton').click();
        // cy.get('.card').eq(1).click();

    });
})