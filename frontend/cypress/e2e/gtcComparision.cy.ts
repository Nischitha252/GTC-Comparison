// describe('WelcomePage', () => {
//     beforeEach(() => {
//         cy.visit('http://localhost:3000');
//     });
//     it('Gets GTC comparision', () => {
//         cy.get('.cardMenu').click();
//         cy.get('.selectPreloadAbbGTC').should('have.text', 'ABB GTC Goods and Services (2020-2 Standard)').click();
//         // cy.get('.card').eq(0).click();
//         // cy.get('.multipleBackButton').click();
//         // cy.get('.card').eq(1).click();
//         // cy.get('.redClauseButton').click();
//         // cy.get('.nonRedClauseButton').click();
//         // cy.get('.downloadButton').click();
//         // cy.get('.commercialBackButton').click();

//     });
// })
// import * as content from '../fixtures/inputData.json';
import 'cypress-file-upload';
describe('WelcomePage', () => {
    beforeEach(() => {
        cy.visit('http://localhost:3000');
    });

    it('Gets GTC comparison', () => {
        cy.get('.cardMenu').click();
        cy.get('.selectPreloadAbbGTC').select('ABB GTC Goods and Services (2020-2 Standard)');
        cy.get('.selectPreloadAbbGTC').select('ABB GTC Goods and Services (2021 Italy)');
        cy.get('.selectPreloadAbbGTC').select('ABB GTC IT Procurement (2023-04)');
        cy.get('.selectPreloadAbbGTC').select('ABB GTC IT Procurement Hardware Schedule (2022-03)');
        cy.get('.selectPreloadAbbGTC').select('ABB GTC IT Procurement Software License Schedule (2023-04)');
        cy.get('.selectPreloadAbbGTC').select('Others (Upload your GTC)');
        cy.get('.selectPreloadAbbGTC').select('ABB GTC Goods and Services (2020-2 Standard)');
        cy.get('.selectNumberOfSupplier').select('1');
        cy.get('.browseFile').click();
        cy.get("input[type=file]").attachFile('test2.pdf');
        cy.get('.uploadButton').click();
        cy.wait(5000);
        cy.get('.allClauseButton').click();
        cy.get('.redClauseButton ').click();
        cy.get('.nonRedClauseButton').click();
        cy.get('.backButton').click();
        cy.get('.cardMenu').click();
        cy.get('.selectPreloadAbbGTC').select('Others (Upload your GTC)');
        cy.get('.browseFile').click();
        cy.get("input[type=file]").attachFile('test2.pdf');
        cy.get('.uploadButton').click();
        cy.wait(5000);
        cy.get('.backButton').click();
        cy.get('.cardMenu').click();
        cy.get('.selectPreloadAbbGTC').select('ABB GTC Goods and Services (2020-2 Standard)');
        cy.get('.selectNumberOfSupplier').select('1');
        cy.get('.browseFile').click();
        cy.get("input[type=file]").attachFile('test2.pdf');
        cy.get('.resetButton').click();

        // cy.get('.selectNumberOfSupplier').select('1');
        // cy.get("input[type=file]").attachFile('1682209 1 1.pdf');
        // cy.get('.resetButton').click();

        // Add further steps based on your test logic
        // For example:
        // cy.get('.card').eq(0).click();
        // cy.get('.multipleBackButton').click();
        // cy.get('.card').eq(1).click();
        // cy.get('.redClauseButton').click();
        // cy.get('.nonRedClauseButton').click();
        // cy.get('.downloadButton').click();
        // cy.get('.commercialBackButton').click();
    });
});



