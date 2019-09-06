#!groovy

@Library('katsdpjenkins') _

katsdp.killOldJobs()
katsdp.setDependencies(['ska-sa/katsdpdockerbase/master',
                        'ska-sa/katdal/master',
                        'ska-sa/katsdpservices/master',
                        'ska-sa/katsdptelstate/master'])
katsdp.standardBuild(python3: true, python2: false)
katsdp.mail('bmerry@ska.ac.za')
